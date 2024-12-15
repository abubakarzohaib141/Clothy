{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/abubakarzohaib141/Clothy/blob/main/Clothy_Bot.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TN89YS7qNe2a",
        "outputId": "b649d68d-35b0-41bc-ab02-f637ecb2fbf4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/411.2 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m411.2/411.2 kB\u001b[0m \u001b[31m25.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/135.8 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m135.8/135.8 kB\u001b[0m \u001b[31m10.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/41.3 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m41.3/41.3 kB\u001b[0m \u001b[31m2.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/40.9 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m40.9/40.9 kB\u001b[0m \u001b[31m2.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "%pip install --quiet -U langchain_core langgraph langchain_google_genai"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 46,
      "metadata": {
        "id": "EnGZhcV9NC0f"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage\n",
        "from langchain_google_genai import ChatGoogleGenerativeAI\n",
        "from langgraph.graph import MessagesState, StateGraph, START, END\n",
        "from langgraph.graph.state import CompiledStateGraph\n",
        "from langgraph.checkpoint.memory import MemorySaver\n",
        "import requests\n",
        "\n",
        "\n",
        "GOOGLE_API_KEY = os.environ.get(\"GEMINI_API_KEY\")\n",
        "LANGCHAIN_API_KEY = os.environ.get(\"LANGCHAIN_API_KEY\")\n",
        "# os.environ[\"LANGCHAIN_API_KEY\"] = \"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\"\n",
        "os.environ[\"LANGCHAIN_TRACING_V2\"] = \"true\"\n",
        "os.environ[\"LANGCHAIN_PROJECT\"] = \"outfit-shop\"\n",
        "\n",
        "model = ChatGoogleGenerativeAI(model=\"gemini-1.5-flash\")\n",
        "\n",
        "# Define the MockAPI Base URL\n",
        "MOCKAPI_URL = \"https://675e967563b05ed0797a7ef4.mockapi.io/products\"\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "# Extend MessagesState for custom chatbot state\n",
        "class ShopState(MessagesState):\n",
        "    summary: str = \"\"  # Summary of the conversation\n",
        "    preferences: dict = {}  # User preferences\n",
        "\n",
        "# Tool-calling functions\n",
        "def fetch_products():\n",
        "    response = requests.get(MOCKAPI_URL)\n",
        "    if response.status_code == 200:\n",
        "        return response.json()\n",
        "    return {\"error\": \"Failed to fetch products\"}\n",
        "\n",
        "def fetch_product_by_id(product_id):\n",
        "    response = requests.get(f\"{MOCKAPI_URL}/{product_id}\")\n",
        "    if response.status_code == 200:\n",
        "        return response.json()\n",
        "    return {\"error\": f\"Failed to fetch product with ID {product_id}\"}\n",
        "\n",
        "def create_product(name, price, size, stock):\n",
        "    response = requests.post(MOCKAPI_URL, json={\"name\": name, \"price\": price, \"size\": size, \"stock\": stock})\n",
        "    if response.status_code == 201:\n",
        "        return response.json()\n",
        "    return {\"error\": \"Failed to create product\"}\n",
        "\n",
        "# LangGraph nodes\n",
        "def call_model(state: ShopState):\n",
        "    \"\"\"Core conversational logic.\"\"\"\n",
        "    summary = state.get(\"summary\", \"\")\n",
        "    system_message = f\"Conversation Summary:\\n{summary}\" if summary else \"You're a helpful assistant for a boys' clothing store.\"\n",
        "    messages = [SystemMessage(content=system_message)] + state[\"messages\"]\n",
        "    response = model.invoke(messages)\n",
        "    return {\"messages\": response}\n",
        "\n",
        "def summarize_conversation(state: ShopState):\n",
        "    \"\"\"Summarize the conversation.\"\"\"\n",
        "    summary = state.get(\"summary\", \"\")\n",
        "    prompt = (\n",
        "        f\"Summarize this conversation about boys' clothing shop:\\n\\n{summary}\" if summary\n",
        "        else \"Summarize this conversation for a boys' clothing shop assistant.\"\n",
        "    )\n",
        "    messages = state[\"messages\"] + [HumanMessage(content=prompt)]\n",
        "    response = model.invoke(messages)\n",
        "    # Retain only the last 2 messages in full detail\n",
        "    delete_messages = [RemoveMessage(id=m.id) for m in state[\"messages\"][:-2]]\n",
        "    return {\"summary\": response.content, \"messages\": delete_messages}\n",
        "\n",
        "def should_continue(state: ShopState):\n",
        "    \"\"\"Determine if we should summarize or continue the conversation.\"\"\"\n",
        "    if len(state[\"messages\"]) > 6:\n",
        "        return \"summarize_conversation\"\n",
        "    return END\n",
        "\n",
        "# Define the workflow\n",
        "workflow = StateGraph(ShopState)\n",
        "\n",
        "# Add nodes\n",
        "workflow.add_node(\"conversation\", call_model)\n",
        "workflow.add_node(\"summarize_conversation\", summarize_conversation)\n",
        "\n",
        "# Define workflow edges\n",
        "workflow.add_edge(START, \"conversation\")\n",
        "workflow.add_conditional_edges(\"conversation\", should_continue)\n",
        "workflow.add_edge(\"summarize_conversation\", END)\n",
        "\n",
        "# Compile the graph with memory\n",
        "memory = MemorySaver()\n",
        "graph = workflow.compile(checkpointer=memory)\n",
        "\n",
        "# Chatbot runtime logic\n",
        "def run_chatbot():\n",
        "    \"\"\"Run the chatbot.\"\"\"\n",
        "    config = {\"configurable\": {\"thread_id\": \"customer-session\"}}\n",
        "\n",
        "    print(\"Welcome to the Boys' Clothing Shop Chatbot! Type 'exit' to end the conversation.\")\n",
        "\n",
        "    while True:\n",
        "        user_input = input(\"You: \")\n",
        "        if user_input.lower() == \"exit\":\n",
        "            print(\"Goodbye! Thank you for visiting.\")\n",
        "            break\n",
        "\n",
        "        # Create a HumanMessage and invoke the graph\n",
        "        input_message = HumanMessage(content=user_input)\n",
        "        output = graph.invoke({\"messages\": [input_message]}, config)\n",
        "\n",
        "        # Get the chatbot's response\n",
        "        bot_response = output[\"messages\"][-1].content\n",
        "        print(f\"Chatbot: {bot_response}\")\n",
        "\n",
        "        # Show updated summary\n",
        "        state = graph.get_state(config)\n",
        "        if \"summary\" in state.values:\n",
        "            print(\"\\nCurrent Summary:\")\n",
        "            print(state.values[\"summary\"])\n",
        "\n",
        "# Run the chatbot\n",
        "run_chatbot()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JF9Dj5kRlWvc",
        "outputId": "0a629844-98b9-4bd4-be9a-292ccd68e4dc"
      },
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Welcome to the Boys' Clothing Shop Chatbot! Type 'exit' to end the conversation.\n",
            "You: I want a blue Shirt\n",
            "Chatbot: Okay, I can help you find a blue shirt! To give you the best recommendations, could you tell me a little more about what you're looking for?  For example:\n",
            "\n",
            "* **What shade of blue?** (e.g., light blue, navy, royal blue, turquoise)\n",
            "* **What style of shirt?** (e.g., t-shirt, polo shirt, button-down shirt, henley)\n",
            "* **What occasion is it for?** (e.g., everyday wear, a special event, school)\n",
            "* **What size are you?**\n",
            "* **What's your budget?**\n",
            "\n",
            "The more information you give me, the better I can assist you in finding the perfect blue shirt!\n",
            "\n",
            "You: exit\n",
            "Goodbye! Thank you for visiting.\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMrv1S0LCX2zxF0wcryI+fs",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}