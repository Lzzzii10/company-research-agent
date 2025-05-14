from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from typing import Dict, Any
from langchain_openai import ChatOpenAI
import os
import logging

logger = logging.getLogger(__name__)

from ..classes import ResearchState
from ..utils.references import format_references_section

class Editor:
    """Compiles individual section briefings into a cohesive final report."""
    
    def __init__(self) -> None:
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Configure a single OpenAI client for all tasks
        self.llm_client = ChatOpenAI(
            model="qwen2.5_72b_instruct-gptq-int4",
            openai_api_key=self.openai_key,
            openai_api_base="http://172.17.3.88:8021/v1",
            temperature=0
        )
        
        # Initialize context dictionary for use across methods
        self.context = {
            "company": "Unknown Company",
            "industry": "Unknown",
            "hq_location": "Unknown"
        }

    async def compile_briefings(self, state: ResearchState) -> ResearchState:
        """Compile individual briefing categories from state into a final report."""
        company = state.get('company', 'Unknown Company')
        
        # Update context with values from state
        self.context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown')
        }
        
        # Send initial compilation status
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message=f"Starting report compilation for {company}",
                    result={
                        "step": "Editor",
                        "substep": "initialization"
                    }
                )

        context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown')
        }
        
        msg = [f"📑 Compiling final report for {company}..."]
        
        # Pull individual briefings from dedicated state keys
        briefing_keys = {
            'company': 'company_briefing',
            'industry': 'industry_briefing',
            'financial': 'financial_briefing',
            'news': 'news_briefing'
        }

        # Send briefing collection status
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message="Collecting section briefings",
                    result={
                        "step": "Editor",
                        "substep": "collecting_briefings"
                    }
                )

        individual_briefings = {}
        for category, key in briefing_keys.items():
            if content := state.get(key):
                individual_briefings[category] = content
                msg.append(f"Found {category} briefing ({len(content)} characters)")
            else:
                msg.append(f"No {category} briefing available")
                logger.error(f"Missing state key: {key}")
        
        if not individual_briefings:
            msg.append("\n⚠️ No briefing sections available to compile")
            logger.error("No briefings found in state")
        else:
            try:
                compiled_report = await self.edit_report(state, individual_briefings, context)
                if not compiled_report or not compiled_report.strip():
                    logger.error("Compiled report is empty!")
                else:
                    logger.info(f"Successfully compiled report with {len(compiled_report)} characters")
            except Exception as e:
                logger.error(f"Error during report compilation: {e}")
        state.setdefault('messages', []).append(AIMessage(content="\n".join(msg)))
        return state
    
    async def edit_report(self, state: ResearchState, briefings: Dict[str, str], context: Dict[str, Any]) -> str:
        """Compile section briefings into a final report and update the state."""
        try:
            company = self.context["company"]
            
            # Step 1: Initial Compilation
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Compiling initial research report",
                        result={
                            "step": "Editor",
                            "substep": "compilation"
                        }
                    )

            edited_report = await self.compile_content(state, briefings, company)
            if not edited_report:
                logger.error("Initial compilation failed")
                return ""

            # Step 2: Deduplication and Cleanup
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Cleaning up and organizing report",
                        result={
                            "step": "Editor",
                            "substep": "cleanup"
                        }
                    )

            # Step 3: Formatting Final Report
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Formatting final report",
                        result={
                            "step": "Editor",
                            "substep": "format"
                        }
                    )
            final_report = await self.content_sweep(state, edited_report, company)
            
            final_report = final_report or ""
            
            logger.info(f"Final report compiled with {len(final_report)} characters")
            if not final_report.strip():
                logger.error("Final report is empty!")
                return ""
            
            logger.info("Final report preview:")
            logger.info(final_report[:500])
            
            # Update state with the final report in two locations
            state['report'] = final_report
            state['status'] = "editor_complete"
            if 'editor' not in state or not isinstance(state['editor'], dict):
                state['editor'] = {}
            state['editor']['report'] = final_report
            logger.info(f"Report length in state: {len(state.get('report', ''))}")
            
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="editor_complete",
                        message="Research report completed",
                        result={
                            "step": "Editor",
                            "report": final_report,
                            "company": company,
                            "is_final": True,
                            "status": "completed"
                        }
                    )
            
            return final_report
        except Exception as e:
            logger.error(f"Error in edit_report: {e}")
            return ""
    
    async def compile_content(self, state: ResearchState, briefings: Dict[str, str], company: str) -> str:
        """Initial compilation of research sections."""
        combined_content = "\n\n".join(content for content in briefings.values())
        
        references = state.get('references', [])
        reference_text = ""
        if references:
            logger.info(f"Found {len(references)} references to add during compilation")
            
            # Get pre-processed reference info from curator
            reference_info = state.get('reference_info', {})
            reference_titles = state.get('reference_titles', {})
            
            logger.info(f"Reference info from state: {reference_info}")
            logger.info(f"Reference titles from state: {reference_titles}")
            
            # Use the references module to format the references section
            reference_text = format_references_section(references, reference_info, reference_titles)
            logger.info(f"Added {len(references)} references during compilation")
        
        # 使用集中上下文的值
        company = self.context["company"]
        industry = self.context["industry"]
        hq_location = self.context["hq_location"]

        prompt = f"""你是一位专业的报告编辑，现在需要将关于{company}的研究简报整合成一份全面的公司研究报告。

已整理的简报内容如下：
{combined_content}

请根据以下要求，撰写一份关于{company}（这是一家总部位于{hq_location}的{industry}公司）的综合性、重点突出的研究报告：

1. 将所有部分的信息整合为连贯且无重复的叙述
2. 保留每个部分的重要细节
3. 合理组织内容，去除过渡性评论或解释
4. 使用清晰的分节标题和结构

格式要求：
严格按照以下文档结构输出（不要更改标题顺序和格式）：

# {company} 研究报告

## 公司概览
[公司内容，包含###子标题]

## 行业概览
[行业内容，包含###子标题]

## 财务概览
[财务内容，包含###子标题]

## 新闻
[新闻内容，包含###子标题]

请以干净的Markdown格式返回报告，不要添加任何解释或评论。"""

        try:
            response = await self.llm_client.ainvoke([
                SystemMessage(content="你是一位专业的报告编辑，负责将研究简报整合成全面的公司研究报告。"),
                HumanMessage(content=prompt)
            ])
            initial_report = response.content.strip()

            # LLM处理后追加参考文献部分
            if reference_text:
                initial_report = f"{initial_report}\n\n{reference_text}"

            return initial_report
        except Exception as e:
            logger.error(f"Error in initial compilation: {e}")
            return (combined_content or "").strip()
        
    async def content_sweep(self, state: ResearchState, content: str, company: str) -> str:
        """Sweep the content for any redundant information."""
        # Use values from centralized context
        company = self.context["company"]
        industry = self.context["industry"]
        hq_location = self.context["hq_location"]
        
        prompt = f"""你是一位专业的报告编辑。你收到了一份关于{company}的研究报告。

当前报告内容如下：
{content}

请严格按照以下要求进行处理：

1. 删除冗余或重复的信息
2. 删除与{company}（一家总部位于{hq_location}的{industry}公司）无关的信息
3. 删除内容空洞、无实质信息的部分
4. 删除所有元评论或过渡性说明（如“以下是新闻...”等）

严格遵循以下文档结构（不要更改标题顺序和格式）：

## 公司概览
[公司内容，包含###子标题]

## 行业概览
[行业内容，包含###子标题]

## 财务概览
[财务内容，包含###子标题]

## 新闻
[新闻内容，仅用*号要点，不要有标题]

## 参考文献
[MLA格式参考文献——务必原样保留，不要更改]

关键规则：
1. 文档必须以“# {company} 研究报告”开头
2. 只允许使用以下##标题，且顺序如下：
   - ## 公司概览
   - ## 行业概览
   - ## 财务概览
   - ## 新闻
   - ## 参考文献
3. 不允许出现其他##标题
4. 公司/行业/财务部分的子标题请用###，新闻部分只用*号要点，不要用标题
5. 严禁出现代码块（```）
6. 各部分之间最多只允许有一个空行
7. 所有要点请用*号格式
8. 每个部分/列表前后请加一个空行
9. 参考文献部分格式必须保持不变，不能修改

请以干净、无解释的Markdown格式返回清理后的报告，不要添加任何说明或评论。
"""
        
        try:
            response_stream = self.llm_client.astream([
                SystemMessage(content="你是一位专业的Markdown格式化编辑，确保文档结构一致且规范。"),
                HumanMessage(content=prompt)
            ])
            
            accumulated_text = ""
            buffer = ""
            
            async for chunk in response_stream:
                chunk_text = chunk.content
                if chunk_text:
                    accumulated_text += chunk_text
                    buffer += chunk_text
                    
                    # Send buffer content via WebSocket if conditions are met
                    if any(char in buffer for char in ['.', '!', '?', '\\n']) and len(buffer) > 10:
                        websocket_manager = state.get('websocket_manager')
                        if websocket_manager:
                            job_id = state.get('job_id')
                            if job_id:
                                await websocket_manager.send_status_update(
                                    job_id=job_id,
                                    status="report_chunk",
                                    message="Formatting final report",
                                    result={
                                        "chunk": buffer,
                                        "step": "Editor"
                                    }
                                )
                        buffer = ""
            # After the loop, send any remaining text in the buffer
            if buffer:
                websocket_manager = state.get('websocket_manager')
                if websocket_manager:
                    job_id = state.get('job_id')
                    if job_id:
                        await websocket_manager.send_status_update(
                            job_id=job_id,
                            status="report_chunk",
                            message="Formatting final report",
                            result={
                                "chunk": buffer,
                                "step": "Editor"
                            }
                        )
            
            return (accumulated_text or "").strip()
        except Exception as e:
            logger.error(f"Error in formatting: {e}")
            return (content or "").strip()

    async def run(self, state: ResearchState) -> ResearchState:
        state = await self.compile_briefings(state)
        # Ensure the Editor node's output is stored both top-level and under "editor"
        if 'report' in state:
            if 'editor' not in state or not isinstance(state['editor'], dict):
                state['editor'] = {}
            state['editor']['report'] = state['report']
        return state
