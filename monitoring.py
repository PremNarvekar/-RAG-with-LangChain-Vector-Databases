import logging
import json
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langsmith import traceable

load_dotenv()


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for log aggregation."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """Set up structured JSON logging."""

    logger = logging.getLogger("LangGraph_App")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics = {
            "request_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):
        self.metrics["request_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> dict:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        error_rate = (
            self.metrics["errors_total"] / self.metrics["request_total"]
            if self.metrics["request_total"] > 0
            else 0
        )

        cache_hit_rate = (
            self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )

        return {
            "total_requests": self.metrics["request_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


class InstrumentedLLM:
    """LLM with full instrumentation."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    @traceable(name="instrumented_invoke")
    def invoke(self, query: str) -> str:
        start_time = time.time()

        try:
            response = self.llm.invoke(query)
            result = response.content

            input_tokens = len(query.split()) * 4 // 3
            output_tokens = len(result.split()) * 4 // 3

            latency = (time.time() - start_time) * 1000

            self.metrics.record_request(
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            self.logger.info(
                "LLM request complete",
                extra={
                    "extra_data": {
                        "latency_ms": latency,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )

            return result

        except Exception as e:
            latency = (time.time() - start_time) * 1000

            self.metrics.record_request(
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            self.logger.error(
                f"LLM request failed: {e}",
                extra={
                    "extra_data": {
                        "error": str(e),
                    }
                },
            )

            raise


def demo_monitoring():
    """Demonstrate monitoring."""

    llm = InstrumentedLLM()

    print("Monitoring Demo\n")

    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    for query in queries:
        result = llm.invoke(query)
        print(f"Query: {query}")
        print(f"Answer: {result[:60]}...\n")

    print("Metrics Summary:\n")

    summary = llm.metrics.get_summary()

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    logger = setup_logging()

    logger.info(
        "Logging setup complete",
        extra={
            "extra_data": {
                "app": "langgraph"
            }
        },
    )

    demo_monitoring()