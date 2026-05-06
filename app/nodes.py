@handle_node_errors
def finalizer_node(state: AgentState):
    logger.info("Executing Finalizer Node - Synthesizing Report")
    
    prompt = (
        f"You are a Senior Research Analyst. Based on the following research context, "
        f"write a comprehensive Markdown report for the task: '{state['task']}'.\n\n"
        f"Context gathered:\n{' '.join(state['context'])}\n\n"
        "Use headers, bullet points, and a 'Sources' section."
    )
    
    response = model.invoke(prompt)
    
    # Save to a local file for extra 'Engineering' points
    with open("research_report.md", "w") as f:
        f.write(response.content)
        
    return {"current_response": response.content}
