#!/usr/bin/env python
"""
Test script for the agent API.
This script tests two kinds of functionality:
1. Simple greeting response
2. Tool usage with a "list nodes" command
"""

import requests
import json
import time
import uuid

# Base URL for the agent server
BASE_URL = "http://localhost:8000/v1/chat/completions"

def make_request(message, conversation_id=None, stream=True):
    """Make a request to the agent server"""
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": message}],
        "stream": stream
    }
    
    # Include conversation_id in query params if provided
    params = {}
    if conversation_id:
        params["conversation_id"] = conversation_id
    
    print(f"\nSending request with message: '{message}'")
    if conversation_id:
        print(f"Using conversation_id: {conversation_id}")
    
    # Make the request
    response = requests.post(BASE_URL, headers=headers, json=payload, params=params, stream=stream)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    if stream:
        # Process the streaming response
        print("\nStreaming response:")
        conversation_content = ""
        tool_usage_detected = False
        
        for line in response.iter_lines():
            if line:
                # Remove 'data: ' prefix and parse JSON
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices'] and 'delta' in data['choices'][0]:
                            delta = data['choices'][0]['delta']
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end='', flush=True)
                                conversation_content += content
                                
                                # Check for tool usage markers
                                if "<tool>" in content:
                                    tool_usage_detected = True
                    except json.JSONDecodeError:
                        print(f"Invalid JSON: {data_str}")
        
        print("\n")
        return {
            "content": conversation_content,
            "tool_usage_detected": tool_usage_detected
        }
    else:
        # Process the non-streaming response
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(f"\nResponse: {content}")
        return {
            "content": content,
            "tool_usage_detected": "<tool>" in content
        }

def test_simple_greeting():
    """Test a simple greeting response"""
    print("\n==== TESTING SIMPLE GREETING ====")
    test_conversation_id = f"test_greeting_{uuid.uuid4().hex[:8]}"
    response = make_request("Hi there!", conversation_id=test_conversation_id)
    
    if response:
        print("\nSIMPLE GREETING TEST RESULT:")
        print(f"Response received: {'✅' if response['content'] else '❌'}")
        print(f"Tool usage detected: {'No ✅' if not response['tool_usage_detected'] else 'Yes ❌'}")
    else:
        print("Test failed - no response received")

def test_node_count():
    """Test tool usage with a node count request"""
    print("\n==== TESTING NODE COUNT ====")
    test_conversation_id = f"test_nodes_{uuid.uuid4().hex[:8]}"
    response = make_request("list the nodes", conversation_id=test_conversation_id)
    
    if response:
        print("\nNODE COUNT TEST RESULT:")
        print(f"Response received: {'✅' if response['content'] else '❌'}")
        print(f"Tool usage detected: {'Yes ✅' if response['tool_usage_detected'] else 'No ❌'}")
    else:
        print("Test failed - no response received")

def main():
    """Run all tests"""
    test_simple_greeting()
    time.sleep(2)
    test_node_count()

if __name__ == "__main__":
    main() 