## [Your Architecture Answer Is Wrong](youtu.be/lYEHDqXagJg)
The speaker highlights that many candidates fail not due to a lack of knowledge, but because they focus on what they built rather than why they made specific architectural decisions .

Key takeaways for experienced engineers:

- Focus on Judgment, Not Completeness: Interviewers are testing your judgment and ability to navigate trade-offs, not just your knowledge of tools

- Demonstrate Production Awareness: Explain your decisions, alternatives considered, failure modes accepted, and mitigation strategies . This shows you've shipped systems and understand real-world production challenges .

- Concrete Example (Memory/RAGrade-offs): The speaker provides an example of how to frame an answer about memory in an agent .  
    - Instead of just saying "I added memory with a vector database," a senior answer would discuss:
    - Choices Made: The decision between full conversation replay and embedding-based retrieval .
    - Trade-offs: Full replay was reliable but had latency and context window issues , while embedding retrieval was cheaper but led to consistency problems due to mixed sessions .

- Decision and Justification: Choosing retrieval with a session-scoped namespace, accepting coherence risk, and setting a hard cap on retrieval count .
- Observability and Risk Monitoring: Monitoring for cases where the cap was hit as a signal for problematic sessions .
- Senior Answer Skeleton: A strong answer should address:
    - The decision made .
    - Alternatives considered and rejected .
    - Failure modes accepted .
    - Mitigations added to reduce failure scenarios .
    - Signals monitored to ensure the solution works as expected .
    - The video encourages viewers to apply this framework to their own projects and share their reframed answers in the comments .


## [Most Engineers Fail These Agentic AI Interview Questions](https://youtu.be/uwropNjYoAs)
1. dont say sued cheaper model to reduce agents' cost
    - break it down into parts and look for issues and then the solution
    - look at cost in all stages, input output, embedding
    - check for redundcancies
    - data storage, indexing cost
    - caching and deduplicating
    - input filtering
    - batching embedding
    - do ab testing with all the scenarios and then switch parts
2. guardrails
    - not only output guardrail
    - prevention is better than cure
    - tool calling constriants rules
