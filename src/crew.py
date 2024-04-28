from crewai import Crew, Process
from src.config.agents import PathwayGeneratorAgent
from src.config.tasks import PathwayGenaratorTask
from langchain_google_genai import GoogleGenerativeAI


def run_crew(input: str):

    agents = PathwayGeneratorAgent()
    tasks = PathwayGenaratorTask()

    # Agents
    pathway_generator_agent = agents.pathway_generator_agent()
    job_finder_agent = agents.job_finder_agent()
    path_finder_agent = agents.path_finder_agent()
    resume_builder = agents.resume_builder()
    final_report_writer = agents.final_report_generator()

    # Tasks
    find_paths = tasks.find_paths(agent=pathway_generator_agent, prompt=input)
    analyse_jobs = tasks.analyse_jobs(agent=job_finder_agent, prompt=input)
    pathway = tasks.create_pathway(
        agent=path_finder_agent, context=[find_paths, analyse_jobs], prompt=input
    )
    current_resume_builder = tasks.create_current_sample_resume(
        resume_builder, prompt=input
    )
    future_resume_builder = tasks.create_future_sample_resume(
        resume_builder, prompt=input, context=[pathway]
    )
    final_report_build = tasks.final_report_compiler(
        final_report_writer,
        prompt=input,
        context=[pathway, current_resume_builder, future_resume_builder],
    )

    # Crew
    crew = Crew(
        agents=[
            pathway_generator_agent,
            job_finder_agent,
            path_finder_agent,
            resume_builder,
            final_report_writer,
        ],
        tasks=[
            find_paths,
            analyse_jobs,
            pathway,
            current_resume_builder,
            future_resume_builder,
            final_report_build,
        ],
        verbose=True,
        process=Process.hierarchical,
        manager_llm=GoogleGenerativeAI(model="gemini-pro"),
    )
    res = crew.kickoff()
    return res
