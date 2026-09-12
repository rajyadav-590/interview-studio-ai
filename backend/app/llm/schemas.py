from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ================================================================
# Interview Evaluation
# ================================================================

class EvaluationResult(BaseModel):

    correctness: int = Field(
        ge=0,
        le=10,
    )

    technical_knowledge: int = Field(
        ge=0,
        le=10,
    )

    relevance: int = Field(
        ge=0,
        le=10,
    )

    clarity: int = Field(
        ge=0,
        le=10,
    )

    communication: int = Field(
        ge=0,
        le=10,
    )

    overall_score: int = Field(
        ge=0,
        le=10,
    )

    strengths: List[str]

    weaknesses: List[str]

    missing_points: List[str]

    improvement_tips: List[str]

    ideal_answer: str

    source: str = "gemini"

    fallback_reason: str | None = None


# ================================================================
# Interview Question
# ================================================================

class QuestionResult(BaseModel):

    question: str

    source: str = "gemini"

    fallback_reason: str | None = None


# ================================================================
# Interview Report
# ================================================================

class ReportResult(BaseModel):

    summary: str

    strongest_skills: List[str]

    skill_gaps: List[str]

    technical_assessment: str

    communication_assessment: str

    problem_solving_assessment: str

    role_readiness: str

    improvement_plan: List[str]

    seven_day_plan: List[str]

    recommendation: str


# ================================================================
# Resume Analysis
# ================================================================

class ResumeAnalysisResult(BaseModel):

    candidate_summary: str

    technical_skills: List[str]

    data_analytics_skills: List[str]

    web_development_skills: List[str]

    tools_and_technologies: List[str]

    education: List[str]

    experience: List[str]

    projects: List[str]

    strengths: List[str]

    skill_gaps: List[str]

    suitable_roles: List[str]

    resume_quality_score: int = Field(
        ge=0,
        le=100,
    )

    resume_quality_assessment: str

    recommended_interview_topics: List[str]

    recommendations: List[str]


# ================================================================
# Job Description Analysis
# ================================================================

class JDAnalysisResult(BaseModel):

    job_role: str

    job_summary: str

    required_skills: List[str]

    preferred_skills: List[str]

    programming_languages: List[str]

    frameworks_and_libraries: List[str]

    databases: List[str]

    tools_and_platforms: List[str]

    responsibilities: List[str]

    soft_skills: List[str]

    experience_requirements: List[str]

    education_requirements: List[str]

    important_interview_topics: List[str]


# ================================================================
# Resume ↔ JD Match
# ================================================================

class JDMatchResult(BaseModel):

    match_score: int = Field(
        ge=0,
        le=100,
    )

    match_assessment: str

    matched_skills: List[str]

    partially_matched_skills: List[str]

    missing_skills: List[str]

    additional_candidate_skills: List[str]

    matched_requirements: List[str]

    unmet_requirements: List[str]

    recommended_interview_topics: List[str]

    preparation_recommendations: List[str]