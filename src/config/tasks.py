from crewai import Task
from textwrap import dedent
from crewai_tools import SerperDevTool
from crewai import Agent


class PathwayGenaratorTask:

    def find_paths(self, agent, prompt: str):
        return Task(
            description=dedent(
                f"""
            Analyse the market trends and suggest potential career paths based on what a student asked. Also find the potential degree, skills that the students need to acquire, potential career paths that the students can explore and much more which can benifit the student

            The paths might include potential degree, skills that the students need to acquire, potential career paths that the students can explore and much more which can benifit the student

            Here's what the student has to say:
            {prompt}
            """
            ),
            expected_output="A list of potential career paths that the student can explore",
            agent=agent,
        )

    def analyse_jobs(self, agent, prompt: str):
        return Task(
            description=dedent(
                f"""
               Analyse the job market and suggest the student jobs based on their current scenario of the student
               Student input: {prompt}
                """
            ),
            expected_output="A list of potential jobs that the student can explore",
            agent=agent,
        )

    def create_pathway(self, agent, prompt, context):
        return Task(
            description=dedent(
                f"""
                Create a detailed report of pathway for the student based on the data provided by Career Path Suggestor and Job Finder

                Student input: {prompt}

                IMPORTANT:
                - The report of the pathway you give should be eligible for the student to explore
                - The report should be based on what steps the student can take in future
                """
            ),
            expected_output="A detailed report of pathway for the student in markdown format",
            context=context,
            agent=agent,
        )

    def create_current_sample_resume(self, agent, prompt):
        return Task(
            description=dedent(
                f"""
                Create a sample resume of the student based on the student's current position

                Student input: {prompt}

                IMPORTANT: The resume should highlight the current set of skills that the student has and also the skills
                """
            ),
            expected_output="A sample resume of the student in markdown format",
            agent=agent,
        )

    def create_future_sample_resume(self, agent, prompt, context):
        return Task(
            description=dedent(
                f"""
                Create a sample resume of the student based on the student's future position if he follows the pathway

                Student input: {prompt}

                IMPORTANT: The resume should highlight the current set of skills that the student has and also the skills
                """
            ),
            expected_output="A sample resume of the student in markdown format",
            agent=agent,
            context=context,
        )

    def final_report_compiler(self, agent, prompt, context):
        return Task(
            description=dedent(
                f"""
                Based on the data given and researched by the other agents, compile a report in markdown format which suits with the student input
                The report should be in the following format given below

                Student input: {prompt}

                INSTRUCTIONS:
                Replace the <...> with the data provided by the other agents

                REPORT FORMAT:
                # Student Analysis

                ## Current Position
                <describe about the current situation of the student>

                ## Sample Resume
                <Write the sample resume for the student>

                # Pathways

                <write the pathways that the student can follow>
                
                ## Final Resume

                <Final resume of what the student will become>
                
                """
            ),
            expected_output="A detailed report of pathway for the student in markdown format",
            agent=agent,
            context=context,
        )