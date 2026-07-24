import httpx

from sda_data.ratelimit import RateLimitedClient, TokenBucket


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_bucket_allows_burst_up_to_capacity():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=1.0, capacity=3, clock=clock)
    for _ in range(3):
        bucket.acquire()
    assert clock.slept == []          # burst within capacity never sleeps


def test_bucket_blocks_when_empty_and_refills():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=2.0, capacity=1, clock=clock)
    bucket.acquire()                  # drains the bucket
    bucket.acquire()                  # must wait for one token at 2/sec
    assert clock.slept == [0.5]


def test_client_applies_bucket_per_request():
    clock = FakeClock()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    client = RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=1.0, capacity=1, clock=clock),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )
    r1 = client.get("https://example.com/a")
    r2 = client.get("https://example.com/b")
    assert r1.status_code == r2.status_code == 200
    assert calls == ["/a", "/b"]
    assert clock.slept == [1.0]       # second request waited for a token


def test_client_sends_user_agent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200)

    client = RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )
    client.get("https://example.com/")
    assert seen["ua"] == "sda-thesis-pipeline/0.1 (+research)"
