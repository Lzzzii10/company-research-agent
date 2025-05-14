import google.generativeai as genai
from typing import Dict, Any, Union, List
import os
import logging
from ..classes import ResearchState
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class Briefing:
    """Creates briefings for each research category and updates the ResearchState."""
    
    def __init__(self) -> None:
        self.max_doc_length = 8000  # Maximum document content length

        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Configure ChatOpenAI
        self.openai_client = ChatOpenAI(
            model="qwen2.5_72b_instruct-gptq-int4",
            openai_api_key=self.openai_key,
            openai_api_base="http://172.17.3.88:8021/v1"
        )

    async def generate_category_briefing(
        self, docs: Union[Dict[str, Any], List[Dict[str, Any]]], 
        category: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        company = context.get('company', 'Unknown')
        industry = context.get('industry', 'Unknown')
        hq_location = context.get('hq_location', 'Unknown')
        logger.info(f"Generating {category} briefing for {company} using {len(docs)} documents")

        # Send category start status
        if websocket_manager := context.get('websocket_manager'):
            if job_id := context.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="briefing_start",
                    message=f"Generating {category} briefing",
                    result={
                        "step": "Briefing",
                        "category": category,
                        "total_docs": len(docs)
                    }
                )

        prompts = {
            'company': f"""请为{company}（一家位于{hq_location}的{industry}公司）撰写一份简明的公司简报。
要求如下：
1. 以"{company}是一家[什么类型公司]，为[客户对象]提供[产品/服务]"开头。
2. 按以下标题和要点结构输出：

### 核心产品/服务
* 列出不同的产品或功能
* 仅包含已证实的技术能力

### 管理团队
* 列出主要管理团队成员
* 包含其职位和专长

### 目标市场
* 列出具体目标客户群体
* 列出已证实的应用场景
* 列出已确认的客户或合作伙伴

### 关键差异化
* 列出独特功能
* 列出已证实的优势

### 商业模式
* 说明产品/服务定价
* 列出分销渠道

3. 每个要点必须是独立、完整的事实
4. 不要出现"未找到信息"或"暂无数据"等字样
5. 只用要点，不要写段落
6. 只输出简报内容，不要有任何解释或评论。""",

            'industry': f"""请为{company}（一家位于{hq_location}的{industry}公司）撰写一份简明的行业简报。
要求如下：
1. 按以下标题和要点结构输出：

### 市场概览
* 说明{company}所处的具体市场细分
* 列出市场规模及年份
* 列出增长率及年份区间

### 直接竞争
* 列出直接竞争对手名称
* 列出具体竞争产品
* 列出市场地位

### 竞争优势
* 列出独特技术特性
* 列出已证实的优势

### 市场挑战
* 列出具体且已证实的挑战

2. 每个要点必须是独立、完整的事实
3. 只用要点，不要写段落
4. 不要出现"未找到信息"或"暂无数据"等字样
5. 只输出简报内容，不要有任何解释。""",

            'financial': f"""请为{company}（一家位于{hq_location}的{industry}公司）撰写一份简明的财务简报。
要求如下：
1. 按以下标题和要点结构输出：

### 融资与投资
* 总融资金额及日期
* 列出每轮融资及日期
* 列出投资人名称

### 收入模式
* 如有，说明产品/服务定价

2. 尽量包含具体数字
3. 只用要点，不要写段落
4. 不要出现"未找到信息"或"暂无数据"等字样
5. 切勿重复同一轮融资。若同月有多轮融资，视为同一轮。
6. 切勿出现融资金额区间。请根据信息判断最合理的具体金额。
7. 只输出简报内容，不要有任何解释或评论。""",

            'news': f"""请为{company}（一家位于{hq_location}的{industry}公司）撰写一份简明的新闻简报。
要求如下：
1. 按以下类别用要点列出：

### 重大公告
* 产品/服务发布
* 新举措

### 合作伙伴关系
* 集成
* 合作

### 认可
* 奖项
* 媒体报道

2. 按时间从新到旧排序
3. 每个要点只描述一条新闻事件
4. 不要出现"未找到信息"或"暂无数据"等字样
5. 不要使用###标题，只用要点
6. 只输出简报内容，不要有任何解释或评论。""",
        }
        
        # Normalize docs to a list of (url, doc) tuples
        items = list(docs.items()) if isinstance(docs, dict) else [
            (doc.get('url', f'doc_{i}'), doc) for i, doc in enumerate(docs)
        ]
        # Sort documents by evaluation score (highest first)
        sorted_items = sorted(
            items, 
            key=lambda x: float(x[1].get('evaluation', {}).get('overall_score', '0')), 
            reverse=True
        )
        
        doc_texts = []
        total_length = 0
        for _ , doc in sorted_items:
            title = doc.get('title', '')
            content = doc.get('raw_content') or doc.get('content', '')
            if len(content) > self.max_doc_length:
                content = content[:self.max_doc_length] + "... [content truncated]"
            doc_entry = f"Title: {title}\n\nContent: {content}"
            if total_length + len(doc_entry) < 120000:  # Keep under limit
                doc_texts.append(doc_entry)
                total_length += len(doc_entry)
            else:
                break
        
        separator = "\n" + "-" * 40 + "\n"
        prompt = f"""{prompts.get(category, f'请根据所提供的文档，为{company}（{industry}行业）撰写一份聚焦、信息丰富且有洞见的研究简报。')}

请分析以下文档，提取关键信息。只需输出简报内容，不要有任何解释或评论：

{separator}{separator.join(doc_texts)}{separator}

"""
        
        try:
            logger.info("Sending prompt to LLM")
            # Use ainvoke for async call, and construct HumanMessage
            response = await self.openai_client.ainvoke([HumanMessage(content=prompt)])
            # Access content from the AIMessage response
            content = response.content.strip()
            if not content:
                logger.error(f"Empty response from LLM for {category} briefing")
                return {'content': ''}

            # Send completion status
            if websocket_manager := context.get('websocket_manager'):
                if job_id := context.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="briefing_complete",
                        message=f"Completed {category} briefing",
                        result={
                            "step": "Briefing",
                            "category": category
                        }
                    )

            return {'content': content}
        except Exception as e:
            logger.error(f"Error generating {category} briefing: {e}")
            return {'content': ''}

    async def create_briefings(self, state: ResearchState) -> ResearchState:
        """Create briefings for all categories in parallel."""
        company = state.get('company', 'Unknown Company')
        websocket_manager = state.get('websocket_manager')
        job_id = state.get('job_id')
        
        # Send initial briefing status
        if websocket_manager and job_id:
            await websocket_manager.send_status_update(
                job_id=job_id,
                status="processing",
                message="Starting research briefings",
                result={"step": "Briefing"}
            )

        context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown'),
            "websocket_manager": websocket_manager,
            "job_id": job_id
        }
        logger.info(f"Creating section briefings for {company}")
        
        # Mapping of curated data fields to briefing categories
        categories = {
            'financial_data': ("financial", "financial_briefing"),
            'news_data': ("news", "news_briefing"),
            'industry_data': ("industry", "industry_briefing"),
            'company_data': ("company", "company_briefing")
        }
        
        briefings = {}

        # Create tasks for parallel processing
        briefing_tasks = []
        for data_field, (cat, briefing_key) in categories.items():
            curated_key = f'curated_{data_field}'
            curated_data = state.get(curated_key, {})
            
            if curated_data:
                logger.info(f"Processing {data_field} with {len(curated_data)} documents")
                
                # Create task for this category
                briefing_tasks.append({
                    'category': cat,
                    'briefing_key': briefing_key,
                    'data_field': data_field,
                    'curated_data': curated_data
                })
            else:
                logger.info(f"No data available for {data_field}")
                state[briefing_key] = ""

        # Process briefings in parallel with rate limiting
        if briefing_tasks:
            # Rate limiting semaphore for LLM API
            briefing_semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent briefings
            
            async def process_briefing(task: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single briefing with rate limiting."""
                async with briefing_semaphore:
                    result = await self.generate_category_briefing(
                        task['curated_data'],
                        task['category'],
                        context
                    )
                    
                    if result['content']:
                        briefings[task['category']] = result['content']
                        state[task['briefing_key']] = result['content']
                        logger.info(f"Completed {task['data_field']} briefing ({len(result['content'])} characters)")
                    else:
                        logger.error(f"Failed to generate briefing for {task['data_field']}")
                        state[task['briefing_key']] = ""
                    
                    return {
                        'category': task['category'],
                        'success': bool(result['content']),
                        'length': len(result['content']) if result['content'] else 0
                    }

            # Process all briefings in parallel
            results = await asyncio.gather(*[
                process_briefing(task) 
                for task in briefing_tasks
            ])
            
            # Log completion statistics
            successful_briefings = sum(1 for r in results if r['success'])
            total_length = sum(r['length'] for r in results)
            logger.info(f"Generated {successful_briefings}/{len(briefing_tasks)} briefings with total length {total_length}")

        state['briefings'] = briefings
        return state

    async def run(self, state: ResearchState) -> ResearchState:
        return await self.create_briefings(state)