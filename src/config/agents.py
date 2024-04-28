from crewai import Agent
from textwrap import dedent
from langchain_google_genai import GoogleGenerativeAI
from crewai_tools import SerperDevTool


class PathwayGeneratorAgent:
    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-pro")
        self.searchInternetTool = SerperDevTool()

    def path_finder_agent(self):
        return Agent(
            role="Carrer Path Suggestor",
            goal=dedent(
                """
                Suggest potential carrer paths based on the following information of students:

                    - Current Scenario
                    - Degree
                    - Skills
                    - Living Place
                    - Expectations
                    - Goals
                """
            ),
            backstory="Have been working with students for more than 30 years. Has high knowledge about potential career paths that sudents can explore",
            allow_delegation=False,
            verbose=True,
            tools=[self.searchInternetTool],
            llm=self.llm,
        )

    def job_finder_agent(self):
        return Agent(
            role="Job Finder",
            goal="Find a job analysing the student profile and job market trends",
            backstory="An experienced professional who has the ability to analyse the current situation of a student and suggest jobs based on the student's profile and job market trends.",
            allow_delegation=False,
            verbose=True,
            tools=[self.searchInternetTool],
            llm=self.llm,
        )

    def resume_builder(self):
        return Agent(
            role="Resume Builder",
            goal="Based on the data create a sample resume of the student",
            backstory="An experienced professional who has the ability to create a sample resume",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

    def pathway_generator_agent(self):
        return Agent(
            role="Generate Pathway",
            goal="Generate a detailed pathway for the student based on the student's profile",
            backstory="An experienced report compiler who can generate a detailed pathway for the student based on the student's profile though the data provided to them",
            allow_delegation=True,
            llm=self.llm,
            verbose=True,
        )

    def final_report_generator(self):
        return Agent(
            role="Report Writer",
            goal="Write detailed report for the student with the pathway he/she can follow and contrast what his/her current position is and what his/her future position will be if the person follows the pathway",
            backstory="An experienced analyser who can analyse student positions and create stunning detailed reports",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )
