"""Unit tests for sentiment analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stock_predictor.data.sentiment import (
    _analyze_text_sentiment,
    fetch_finviz_sentiment,
    fetch_reddit_sentiment,
    get_sentiment_features,
    get_sentiment_summary,
    get_trending_tickers_from_social,
)


class TestAnalyzeTextSentiment:
    def test_positive_text(self):
        result = _analyze_text_sentiment("This stock is amazing and will go up!")
        assert result["polarity"] > 0

    def test_negative_text(self):
        result = _analyze_text_sentiment("This stock is terrible, avoid it.")
        assert result["polarity"] < 0

    def test_neutral_text(self):
        result = _analyze_text_sentiment("The company reported earnings.")
        assert -0.5 < result["polarity"] < 0.5

    def test_returns_polarity_and_subjectivity(self):
        result = _analyze_text_sentiment("Hello world")
        assert "polarity" in result
        assert "subjectivity" in result

    def test_empty_text(self):
        result = _analyze_text_sentiment("")
        assert result["polarity"] == 0.0
        assert result["subjectivity"] == 0.0


class TestFetchRedditSentiment:
    _REDDIT_HTML = """<html><body>
    <div class="thing" data-fullname="t3_abc123">
        <div class="score unvoted" title="100">100</div>
        <a class="title" href="/r/wallstreetbets/test">NVDA is going to the moon!</a>
        <a class="comments" href="/r/wallstreetbets/comments/abc123">50 comments</a>
        <time datetime="2023-11-14T12:00:00+00:00">1 day ago</time>
    </div>
    </body></html>"""

    @patch("stock_predictor.data.sentiment._REQUEST_DELAY", 0)
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_parses_reddit_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self._REDDIT_HTML
        mock_get.return_value = mock_response

        posts = fetch_reddit_sentiment("NVDA", limit=5)
        assert len(posts) > 0
        assert posts[0]["source"] == "reddit"
        assert "polarity" in posts[0]
        assert "score" in posts[0]

    @patch("stock_predictor.data.sentiment._REQUEST_DELAY", 0)
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_handles_api_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        posts = fetch_reddit_sentiment("AAPL")
        assert posts == []

    @patch("stock_predictor.data.sentiment._REQUEST_DELAY", 0)
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        posts = fetch_reddit_sentiment("AAPL")
        assert posts == []


class TestGetSentimentFeatures:
    @patch("stock_predictor.data.earnings_transcript.fetch_earnings_transcript")
    @patch("stock_predictor.data.sentiment.fetch_finviz_sentiment")
    @patch("stock_predictor.data.sentiment.fetch_reddit_sentiment")
    def test_aggregates_features(self, mock_reddit, mock_finviz, mock_transcript):
        mock_reddit.return_value = [
            {"polarity": 0.5, "subjectivity": 0.7, "score": 100, "num_comments": 50},
            {"polarity": -0.1, "subjectivity": 0.4, "score": 20, "num_comments": 10},
        ]
        mock_finviz.return_value = [
            {"polarity": 0.3, "subjectivity": 0.5},
        ]
        mock_transcript.return_value = {
            "transcript_sentiment": 0.1,
            "transcript_polarity": 0.05,
            "transcript_url": "https://example.com",
            "transcript_date": "2025-01-30",
            "transcript_text_preview": "Test transcript",
            "transcript_source_texts": [("Earnings Call", "test", 0.05)],
        }

        features = get_sentiment_features("AAPL")
        assert "sentiment_mean_polarity" in features
        assert "reddit_mention_count" in features
        assert features["reddit_mention_count"] == 2
        assert features["finviz_mention_count"] == 1
        assert features["sentiment_total_mentions"] == 4
        assert features["transcript_sentiment"] == 0.1
        assert features["transcript_polarity"] == 0.05
        assert "source_texts" in features

    @patch("stock_predictor.data.earnings_transcript.fetch_earnings_transcript")
    @patch("stock_predictor.data.sentiment.fetch_finviz_sentiment")
    @patch("stock_predictor.data.sentiment.fetch_reddit_sentiment")
    def test_handles_empty_data(self, mock_reddit, mock_finviz, mock_transcript):
        mock_reddit.return_value = []
        mock_finviz.return_value = []
        mock_transcript.return_value = {
            "transcript_sentiment": None,
            "transcript_polarity": None,
            "transcript_url": None,
            "transcript_date": None,
            "transcript_text_preview": None,
            "transcript_source_texts": [],
        }

        features = get_sentiment_features("XXXX")
        assert features["sentiment_mean_polarity"] == 0.0
        assert features["sentiment_total_mentions"] == 0
        assert features["transcript_sentiment"] is None


class TestGetSentimentSummary:
    @patch("stock_predictor.data.sentiment.get_sentiment_features")
    def test_positive_summary(self, mock_features):
        mock_features.return_value = {
            "sentiment_mean_polarity": 0.3,
            "sentiment_total_mentions": 50,
            "reddit_mention_count": 20,
            "reddit_mean_polarity": 0.25,
            "reddit_mean_score": 100.0,
            "finviz_mention_count": 15,
            "finviz_mean_polarity": 0.35,
            "transcript_sentiment": 0.1,
            "transcript_url": "https://example.com",
        }
        summary = get_sentiment_summary("NVDA")
        assert "Strongly Positive" in summary
        assert "Earnings Call" in summary
        assert "NVDA" in summary


class TestGetTrendingTickers:
    @patch("stock_predictor.data.sentiment._REQUEST_DELAY", 0)
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_returns_list(self, mock_get):
        html = """<html><body>
        <div class="thing" data-fullname="t3_1">
            <a class="title" href="/test">NVDA to the moon!</a>
        </div>
        <div class="thing" data-fullname="t3_2">
            <a class="title" href="/test">TSLA breaking out</a>
        </div>
        </body></html>"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        tickers = get_trending_tickers_from_social()
        assert isinstance(tickers, list)
