from Chain import Chain, Model, Prompt, Parser
from pydantic import BaseModel
from typing import List, Dict, Any
import json


# Define our Pydantic models for structured outputs
class SentimentAnalysis(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_emotions: List[str]


class ContentMetadata(BaseModel):
    title: str
    word_count: int
    reading_time: int
    domain: str


class AnalysisReport(BaseModel):
    url: str
    summary: str
    sentiment: SentimentAnalysis
    tags: List[str]
    metadata: ContentMetadata
    analysis_depth: str


class ContentAnalysisPipeline:
    """
    This class recreates the Prompt Flow workflow using Chain objects.
    Each step corresponds to a node in the flow.dag.yaml
    """

    def __init__(self):
        # Initialize our models for different steps
        self.summary_model = Model("o4-mini")
        self.sentiment_model = Model("haiku")
        self.tagging_model = Model("o4-mini")

        # Initialize our prompts (these would correspond to .jinja2 files)
        self.summary_prompt = Prompt(
            """
        Analyze this content and provide a {{analysis_depth}} summary:
        
        Content: {{content}}
        
        Summary guidelines:
        - If analysis_depth is "quick": 1-2 sentences
        - If analysis_depth is "standard": 1 paragraph  
        - If analysis_depth is "detailed": 2-3 paragraphs with key points
        """
        )

        self.sentiment_prompt = Prompt(
            """
        Analyze the sentiment of this content summary:
        
        Summary: {{content}}
        
        Provide your analysis in this exact format:
        - Sentiment: [positive/negative/neutral]
        - Confidence: [0.0-1.0]
        - Key emotions: [list of emotions detected]
        """
        )

        self.tagging_prompt = Prompt(
            """
        Generate relevant tags for this content:
        
        Summary: {{content}}
        Original snippet: {{original_content[:500]}}
        
        Generate 5-8 relevant tags that capture:
        - Main topics
        - Content type
        - Key themes
        - Target audience
        
        Return as a JSON list of strings.
        """
        )

        # Initialize parsers for structured output
        self.sentiment_parser = Parser(SentimentAnalysis)

    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Step 1: Extract content from URL (equivalent to extract_content node)
        This would use your existing Python extraction logic
        """
        # This is a simplified version - you'd implement actual web scraping
        return {
            "text": f"Sample content extracted from {url}...",
            "metadata": {
                "title": "Sample Article",
                "word_count": 1500,
                "reading_time": 6,
                "domain": url.split("//")[1].split("/")[0] if "//" in url else url,
            },
        }

    def generate_summary(self, content: str, analysis_depth: str) -> str:
        """
        Step 2: Generate summary (equivalent to generate_summary node)
        """
        chain = Chain(prompt=self.summary_prompt, model=self.summary_model)

        response = chain.run(
            input_variables={"content": content, "analysis_depth": analysis_depth}
        )

        return response.content

    def analyze_sentiment(self, content: str) -> SentimentAnalysis:
        """
        Step 3: Analyze sentiment (equivalent to analyze_sentiment node)
        """
        chain = Chain(
            prompt=self.sentiment_prompt,
            model=self.sentiment_model,
            parser=self.sentiment_parser,
        )

        response = chain.run(input_variables={"content": content})

        return response.content

    def generate_tags(self, summary: str, original_content: str) -> List[str]:
        """
        Step 4: Generate tags (equivalent to generate_tags node)
        """
        chain = Chain(prompt=self.tagging_prompt, model=self.tagging_model)

        response = chain.run(
            input_variables={"content": summary, "original_content": original_content}
        )

        # Parse JSON response to get list of tags
        try:
            return json.loads(response.content)
        except:
            # Fallback parsing if JSON fails
            return response.content.split(", ")

    def compile_report(
        self,
        url: str,
        summary: str,
        sentiment: SentimentAnalysis,
        tags: List[str],
        metadata: Dict[str, Any],
        analysis_depth: str,
    ) -> AnalysisReport:
        """
        Step 5: Compile final report (equivalent to compile_report node)
        """
        return AnalysisReport(
            url=url,
            summary=summary,
            sentiment=sentiment,
            tags=tags,
            metadata=ContentMetadata(**metadata),
            analysis_depth=analysis_depth,
        )

    def run_pipeline(
        self, url: str, analysis_depth: str = "standard"
    ) -> AnalysisReport:
        """
        Main pipeline execution - orchestrates all the Chain calls
        This recreates the entire Prompt Flow workflow
        """
        # Step 1: Extract content (equivalent to extract_content node)
        extracted = self.extract_content(url)

        # Step 2: Generate summary (equivalent to generate_summary node)
        summary = self.generate_summary(extracted["text"], analysis_depth)

        # Step 3: Analyze sentiment (equivalent to analyze_sentiment node)
        sentiment = self.analyze_sentiment(summary)

        # Step 4: Generate tags (equivalent to generate_tags node)
        tags = self.generate_tags(summary, extracted["text"])

        # Step 5: Compile report (equivalent to compile_report node)
        report = self.compile_report(
            url=url,
            summary=summary,
            sentiment=sentiment,
            tags=tags,
            metadata=extracted["metadata"],
            analysis_depth=analysis_depth,
        )

        return report


# Usage example
if __name__ == "__main__":
    pipeline = ContentAnalysisPipeline()

    # Run the complete pipeline
    result = pipeline.run_pipeline(
        url="https://www.github.com/acesanderson/Chain", analysis_depth="standard"
    )

    print(f"Summary: {result.summary}")
    print(
        f"Sentiment: {result.sentiment.sentiment} (confidence: {result.sentiment.confidence})"
    )
    print(f"Tags: {', '.join(result.tags)}")
    print(f"Report: {result}")
