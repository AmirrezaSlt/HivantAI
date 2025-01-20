import json
import requests
from typing import Dict, Any, Optional, List
from .base import BaseContentProvider
from ...agent.connections import NotionConnection

class NotionContentProvider(BaseContentProvider):
    def __init__(self, workspace: Optional[str] = None, api_key: Optional[str] = None):
        self.connection = NotionConnection(workspace, api_key)

    def get_content(self, block_id: str) -> Dict[str, Any]:
        response = requests.get(f"{self.connection.base_url}/blocks/{block_id}/children", headers=self.connection.headers)
        response.raise_for_status()
        return self.parse_block_content(response.json())

    def parse_block_content(self, content: Dict[str, Any]) -> str:
        parsed_content = {
            'results': []
        }
        for block in content.get('results', []):
            block_type = block.get('type')
            parsed_block = {
                'type': block_type,
            }
            
            if block_type == 'paragraph':
                text = ''.join([rt.get('plain_text', '') for rt in block['paragraph']['rich_text']])
                if text.strip():  # Only include non-empty paragraphs
                    parsed_block['text'] = text
            elif block_type == 'heading_1':
                parsed_block['text'] = ''.join([rt.get('plain_text', '') for rt in block['heading_1']['rich_text']])
            elif block_type == 'heading_2':
                parsed_block['text'] = ''.join([rt.get('plain_text', '') for rt in block['heading_2']['rich_text']])
            elif block_type == 'heading_3':
                parsed_block['text'] = ''.join([rt.get('plain_text', '') for rt in block['heading_3']['rich_text']])
            elif block_type == 'child_page':
                parsed_block['title'] = block['child_page']['title']
            elif block_type == 'image':
                parsed_block['url'] = block['image']['file']['url']
            elif block_type == 'code':
                # Concatenate all rich_text parts to handle multi-line code blocks
                parsed_block['code'] = '\n'.join([rt.get('plain_text', '') for rt in block['code']['rich_text']])
                parsed_block['language'] = block['code']['language']
            elif block_type == 'file':
                parsed_block['url'] = block['file']['file']['url']
            
            if (
                parsed_block.get('text') or 
                parsed_block.get('title') or 
                parsed_block.get('url') or 
                parsed_block.get('code')
            ):
                parsed_content['results'].append(parsed_block)
        
        return json.dumps(parsed_content)

    def list_blocks(self, block_id: str, depth: int = 0) -> List[Dict[str, Any]]:
        content = self.get_content(block_id)
        blocks = []
        
        for block in content.get('results', []):
            block_type = block.get('type')
            block_info = {
                'id': block.get('id'),
                'type': block_type,
                'depth': depth,
            }

            if block_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3']:
                block_info['text'] = block.get('text', '')
            elif block_type == 'child_page':
                block_info['title'] = block.get('title', '')
            elif block_type == 'code':
                block_info['code'] = block.get('code', '')
                block_info['language'] = block.get('language', '')
            elif block_type == 'file':
                block_info['file_url'] = block.get('file_url', '')

            if 'text' in block_info or 'title' in block_info or 'code' in block_info or 'file_url' in block_info:
                blocks.append(block_info)

            if block.get('has_children'):
                blocks.extend(self.list_blocks(block['id'], depth + 1))

        return blocks

    def get_resource_link(self, uri: str) -> str:
        return f"https://notion.so/{self.connection.workspace}/{uri.replace('-', '')}"