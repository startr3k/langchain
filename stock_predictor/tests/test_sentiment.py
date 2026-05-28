"""Unit tests for sentiment analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stock_predictor.data.sentiment import (
    _analyze_text_sentiment,
    fetch_finviz_sentiment,
    fetch_reddit_sentiment,
    fetch_stocktwits_sentiment,
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
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_parses_reddit_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "NVDA is going to the moon!",
                            "selftext": "Buy buy buy",
                            "score": 100,
                            "num_comments": 50,
                            "upvote_ratio": 0.9,
                            "created_utc": 1700000000,
                        }
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        posts = fetch_reddit_sentiment("NVDA", limit=5)
        assert len(posts) > 0
        assert posts[0]["source"] == "reddit"
        assert "polarity" in posts[0]
        assert "score" in posts[0]

    @patch("stock_predictor.data.sentiment.requests.get")
    def test_handles_api_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        posts = fetch_reddit_sentiment("AAPL")
        assert posts == []

    @patch("stock_predictor.data.sentiment.requests.get")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        posts = fetch_reddit_sentiment("AAPL")
        assert posts == []


class TestFetchStockTwitsSentiment:
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_parses_stocktwits_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {
                    "body": "Bullish on TSLA long term",
                    "entities": {"sentiment": {"basic": "Bullish"}},
                    "likes": {"total": 5},
                }
            ]
        }
        mock_get.return_value = mock_response

        msgs = fetch_stocktwits_sentiment("TSLA")
        assert len(msgs) == 1
        assert msgs[0]["source"] == "stocktwits"
        assert msgs[0]["stocktwits_sentiment"] == "Bullish"

    @patch("stock_predictor.data.sentiment.requests.get")
    def test_handles_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        msgs = fetch_stocktwits_sentiment("INVALID")
        assert msgs == []


class TestGetSentimentFeatures:
    @patch("stock_predictor.data.sentiment.fetch_stocktwits_sentiment")
    @patch("stock_predictor.data.sentiment.fetch_finviz_sentiment")
    @patch("stock_predictor.data.sentiment.fetch_reddit_sentiment")
    def test_aggregates_features(self, mock_reddit, mock_finviz, mock_stocktwits):
        mock_reddit.return_value = [
            {"polarity": 0.5, "subjectivity": 0.7, "score": 100, "num_comments": 50},
            {"polarity": -0.1, "subjectivity": 0.4, "score": 20, "num_comments": 10},
        ]
        mock_finviz.return_value = [
            {"polarity": 0.3, "subjectivity": 0.5},
        ]
        mock_stocktwits.return_value = [
            {"polarity": 0.2, "subjectivity": 0.6, "stocktwits_sentiment": "Bullish"},
            {"polarity": -0.3, "subjectivity": 0.8, "stocktwits_sentiment": "Bearish"},
        ]

        features = get_sentiment_features("AAPL")
        assert "sentiment_mean_polarity" in features
        assert "reddit_mention_count" in features
        assert features["reddit_mention_count"] == 2
        assert features["finviz_mention_count"] == 1
        assert features["stocktwits_mention_count"] == 2
        assert features["stocktwits_bullish_count"] == 1
        assert features["stocktwits_bearish_count"] == 1
        assert features["sentiment_total_mentions"] == 5

    @patch("stock_predictor.data.sentiment.fetch_stocktwits_sentiment")
    @patch("stock_predictor.data.sentiment.fetch_finviz_sentiment")
    @patch("stock_predictor.data.sentiment.fetch_reddit_sentiment")
    def test_handles_empty_data(self, mock_reddit, mock_finviz, mock_stocktwits):
        mock_reddit.return_value = []
        mock_finviz.return_value = []
        mock_stocktwits.return_value = []

        features = get_sentiment_features("XXXX")
        assert features["sentiment_mean_polarity"] == 0.0
        assert features["sentiment_total_mentions"] == 0


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
            "stocktwits_mention_count": 15,
            "stocktwits_bullish_count": 10,
            "stocktwits_bearish_count": 3,
            "stocktwits_bull_bear_ratio": 3.33,
        }
        summary = get_sentiment_summary("NVDA")
        assert "Strongly Positive" in summary
        assert "NVDA" in summary


class TestGetTrendingTickers:
    @patch("stock_predictor.data.sentiment.requests.get")
    def test_returns_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {"data": {"title": "NVDA to the moon!", "selftext": ""}},
                    {"data": {"title": "TSLA breaking out", "selftext": "TSLA calls"}},
                ]
            }
        }
        mock_get.return_value = mock_response

        tickers = get_trending_tickers_from_social()
        assert isinstance(tickers, list)
