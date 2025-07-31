#!/usr/bin/env python3
"""
Simplified DeepResearcher Example

This script demonstrates the core functionality of the DeepResearcher agent.
The main logic is: initialize → research → display results → save to file
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentdemo.agents import DeepResearcher


def main():
    """Simple example of DeepResearcher core functionality with file output."""

    # 1. Check environment setup
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set OPENROUTER_API_KEY and GEMINI_API_KEY in your .env file")
        return

    # 2. Initialize the researcher
    print("🚀 Initializing DeepResearcher...")
    researcher = DeepResearcher()

    # 3. Define research topic
    topic = "How does climate change affect coffee production globally?"
    print(f"🔍 Researching: {topic}")

    # 4. Perform research (this is the main logic)
    print("🔄 Starting research...")
    result = researcher.research(user_message=topic)

    # 5. Display results
    print(f"\n✅ Research completed!")
    print(f"📄 Summary: {result.summary}")
    print(f"📊 Sources found: {len(result.sources)}")
    print(f"📖 Content: {result.content}")

    # 6. Show first few sources
    if result.sources:
        print(f"\n🔗 Top sources:")
        for i, source in enumerate(result.sources[:3], 1):
            source_info = source.get("label", source.get("title", "Unknown Source"))
            source_url = source.get("value", "")
            print(f"   {i}. {source_info}")
            if source_url:
                print(f"      URL: {source_url}")

    print(f"\n🎉 Research complete!")


if __name__ == "__main__":
    main()
