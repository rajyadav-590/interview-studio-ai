from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

from ..interview.question_bank import find_question, find_similar_question, questions_for_domain

from .prompts import (
    EVALUATION_SYSTEM_PROMPT,
    QUESTION_SYSTEM_PROMPT,
    evaluation_prompt,
    question_prompt,
    JD_ANALYSIS_SYSTEM_PROMPT,
    JD_MATCHING_SYSTEM_PROMPT,
)

from .schemas import (
    EvaluationResult,
    QuestionResult,
    ReportResult,
    ResumeAnalysisResult,
    JDAnalysisResult,
    JDMatchResult,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# -------------------------------------------------------------------
# Final interview report prompt
# -------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """
You are a senior technical interviewer and career assessment expert.

Analyze the complete interview performance of a candidate.

Your assessment must be based ONLY on:
- the interview role
- experience level
- interview type
- difficulty
- the questions asked
- the candidate's answers
- the numerical evaluations already produced for those answers

Do not invent skills, experience, projects, technologies, or achievements
that are not supported by the interview data.

The purpose is to provide a useful and realistic final assessment.

Requirements:

1. Identify the candidate's strongest demonstrated skills.
2. Identify concrete skill gaps.
3. Evaluate technical knowledge.
4. Evaluate communication.
5. Evaluate problem-solving ability when supported by the answers.
6. Assess readiness for the selected role at the selected experience level.
7. Give practical improvement recommendations.
8. Create a realistic 7-day preparation plan.
9. Keep the assessment constructive but honest.
10. Do not over-praise weak answers.
11. Do not make claims about personality, confidence, intelligence,
    nervousness, or emotions unless explicitly supported by available data.
12. Return ONLY the requested structured JSON.
""".strip()


# -------------------------------------------------------------------
# Resume analysis prompt
# -------------------------------------------------------------------

RESUME_ANALYSIS_SYSTEM_PROMPT = """
You are an expert technical recruiter, resume analyst, and career advisor.

Your task is to analyze a candidate's resume and produce a structured
candidate profile.

IMPORTANT RULES:

1. Analyze ONLY information explicitly supported by the resume text.
2. Never invent skills, technologies, projects, internships, job titles,
   certifications, achievements, or experience.
3. Do not assume that mentioning a technology means the candidate has
   advanced expertise in it.
4. Distinguish between demonstrated skills and merely mentioned skills.
5. Identify technical skills from the resume accurately.
6. Identify data analytics skills separately when supported.
7. Identify web development skills separately when supported.
8. Identify tools and technologies separately.
9. Extract education accurately.
10. Extract internship/work experience accurately.
11. Extract projects accurately.
12. Identify strengths based on evidence in the resume.
13. Identify skill gaps conservatively. A skill gap means an important
    area that appears missing or insufficiently demonstrated for the
    candidate's likely target roles.
14. Suggest suitable entry-level or junior roles based on the actual
    resume content.
15. Give a realistic resume quality score from 0 to 100.
16. Explain the resume quality score clearly.
17. Recommend interview topics based on the actual skills and projects
    present in the resume.
18. Give actionable recommendations for improving the candidate's profile.
19. Do not judge personality, intelligence, confidence, emotions,
    nervousness, or character.
20. Do not make unsupported claims about employment eligibility or hiring.
21. Do not expose private information beyond what is already provided
    in the resume.
22. Return ONLY the structured JSON requested by the schema.

The analysis should be useful for a technical interview preparation system.
""".strip()


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


# -------------------------------------------------------------------
# LLM Service
# -------------------------------------------------------------------

class LLMService:

    def __init__(self) -> None:

        self.api_key = os.getenv(
            "GEMINI_API_KEY",
            "",
        ).strip()

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ).strip()

        self.client: Optional[genai.Client] = None

        if self.api_key:

            try:

                self.client = genai.Client(
                    api_key=self.api_key
                )

                logger.info(
                    "Gemini client initialized successfully using model %s",
                    self.model,
                )

            except Exception:

                logger.exception(
                    "Failed to initialize Gemini client"
                )

                self.client = None

    # ----------------------------------------------------------------
    # Status
    # ----------------------------------------------------------------

    @property
    def configured(self) -> bool:

        return (
            self.client is not None
            and bool(self.api_key)
        )

    @property
    def mode(self) -> str:

        return (
            "gemini"
            if self.configured
            else "mock-fallback"
        )

    # ----------------------------------------------------------------
    # Generic structured Gemini generation
    # ----------------------------------------------------------------

    def _generate_structured(
        self,
        system_instruction: str,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> Optional[BaseModel]:

        if not self.configured or self.client is None:
            return None

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            text = getattr(response, "text", None)

            if not text:
                logger.warning("Gemini returned an empty response")
                return None

            parsed = json.loads(text)
            return response_schema.model_validate(parsed)

        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                logger.warning(
                    "Gemini quota/rate limit reached; using local fallback."
                )
            else:
                logger.exception(
                    "Gemini structured generation failed: %s",
                    exc,
                )
            return None

    def _generate_json_without_schema(
        self,
        system_instruction: str,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> Optional[BaseModel]:
        """
        Fallback JSON generation used when Gemini's native structured
        response-schema generation fails for a particular request.
        The returned JSON is still validated with the Pydantic schema.
        """

        if not self.configured or self.client is None:
            return None

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

            text = getattr(response, "text", None)
            if not text:
                logger.warning("Gemini fallback JSON response was empty")
                return None

            # Be tolerant if the model wraps JSON in a markdown code block.
            cleaned = text.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            parsed = json.loads(cleaned)
            return response_schema.model_validate(parsed)

        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                logger.warning(
                    "Gemini quota/rate limit reached during JSON fallback."
                )
            else:
                logger.exception(
                    "Gemini fallback JSON generation failed: %s",
                    exc,
                )
            return None

    # ----------------------------------------------------------------
    # Resume Intelligence
    # ----------------------------------------------------------------

    @staticmethod
    def _local_resume_analysis(resume_text: str) -> ResumeAnalysisResult:
        """Deterministic resume analyzer for offline/quota-limited demos."""
        text = _clean_text(resume_text)
        lower = text.lower()

        skill_groups = {
            "technical_skills": [
                "python", "java", "c", "c++", "javascript", "typescript", "sql",
                "machine learning", "deep learning", "artificial intelligence", "nlp",
                "computer vision", "data structures", "algorithms", "rest api", "git",
                "github", "docker", "linux", "aws", "azure", "gcp",
            ],
            "data_analytics_skills": [
                "pandas", "numpy", "matplotlib", "seaborn", "power bi", "tableau",
                "excel", "statistics", "data analysis", "data analytics", "scikit-learn",
            ],
            "web_development_skills": [
                "html", "css", "javascript", "react", "node.js", "nodejs", "express",
                "fastapi", "django", "flask", "bootstrap", "tailwind",
            ],
            "tools_and_technologies": [
                "mysql", "postgresql", "mongodb", "sqlite", "firebase", "git", "github",
                "figma", "jupyter", "vs code", "visual studio", "postman", "power bi",
            ],
        }

        def found(term: str) -> bool:
            aliases = {"node.js": ["node.js", "nodejs", "node"], "c++": ["c++"]}
            return any(a in lower for a in aliases.get(term, [term]))

        def nice(term: str) -> str:
            names = {"node.js": "Node.js", "nodejs": "Node.js", "c++": "C++", "c": "C",
                     "sql": "SQL", "html": "HTML", "css": "CSS", "javascript": "JavaScript",
                     "typescript": "TypeScript", "python": "Python", "java": "Java", "git": "Git",
                     "github": "GitHub", "mysql": "MySQL", "postgresql": "PostgreSQL",
                     "mongodb": "MongoDB", "sqlite": "SQLite", "fastapi": "FastAPI",
                     "django": "Django", "flask": "Flask", "react": "React", "pandas": "Pandas",
                     "numpy": "NumPy", "scikit-learn": "scikit-learn", "power bi": "Power BI",
                     "vs code": "VS Code"}
            return names.get(term, term.title())

        result = {}
        for group, terms in skill_groups.items():
            result[group] = []
            for term in terms:
                if found(term) and nice(term) not in result[group]:
                    result[group].append(nice(term))

        lines = [re.sub(r"\s+", " ", x).strip(" -•|\t") for x in text.splitlines()]
        lines = [x for x in lines if x]

        def section_items(names):
            items=[]; active=False
            for line in lines:
                l=line.lower().strip().rstrip(":")
                if any(n in l for n in names):
                    active=True; continue
                if active:
                    if re.match(r"^(skills?|education|experience|work experience|projects?|certifications?|summary|objective|achievements?|interests?)\\s*:??$", l):
                        break
                    if len(line) >= 8:
                        items.append(line)
                    if len(items) >= 8: break
            return items

        education = section_items(["education", "academic background", "qualification"])
        experience = section_items(["experience", "work experience", "internship", "employment"])
        projects = section_items(["projects", "project experience"])
        if not projects:
            projects = [line for line in lines if any(k in line.lower() for k in ["project", "developed ", "built ", "implemented "])][:6]

        candidate_summary = next((x for x in lines if 30 <= len(x) <= 300 and not x.lower().startswith(("resume", "curriculum", "email", "phone"))), "Resume profile extracted from uploaded document.")
        strengths=[]
        if result["technical_skills"]: strengths.append("Technical exposure to " + ", ".join(result["technical_skills"][:5]) + ".")
        if projects: strengths.append("Projects or practical work are documented in the resume.")
        if experience: strengths.append("Experience or internship information is present.")
        if result["data_analytics_skills"]: strengths.append("Data and analytics tools are represented.")
        if not strengths: strengths.append("The resume contains extractable candidate information.")

        likely_roles=[]
        role_map=[
            ("AI/ML Engineer", ["machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn"]),
            ("Python Developer", ["python", "fastapi", "django", "flask"]),
            ("Data Analyst", ["pandas", "power bi", "tableau", "excel", "data analysis"]),
            ("Web Developer", ["html", "css", "javascript", "react", "node.js", "nodejs"]),
            ("Software Engineer", ["python", "java", "c++", "data structures", "algorithms"]),
        ]
        for role, terms in role_map:
            if sum(1 for t in terms if t in lower) >= 2:
                likely_roles.append(role)
        if not likely_roles: likely_roles=["Software Engineer"]

        topics=(result["technical_skills"]+result["data_analytics_skills"]+result["web_development_skills"])[:10]
        gaps=[]
        if "sql" not in lower: gaps.append("SQL/database querying is not clearly demonstrated.")
        if "git" not in lower: gaps.append("Version control with Git is not clearly demonstrated.")
        if "testing" not in lower: gaps.append("Software testing practices are not clearly demonstrated.")
        score=45
        score += min(20, len(result["technical_skills"])*3)
        score += min(12, len(projects)*3)
        score += min(10, len(experience)*3)
        score += min(8, len(result["data_analytics_skills"])+len(result["web_development_skills"]))
        if education: score += 5
        score=max(0,min(100,score))
        recommendations=[
            "Quantify project outcomes with measurable results where the resume supports them.",
            "Keep skills grouped by category and aligned with the target role.",
            "Add concrete technologies and responsibilities to project descriptions where supported.",
        ]
        return ResumeAnalysisResult(
            candidate_summary=candidate_summary,
            technical_skills=result["technical_skills"],
            data_analytics_skills=result["data_analytics_skills"],
            web_development_skills=result["web_development_skills"],
            tools_and_technologies=result["tools_and_technologies"],
            education=education,
            experience=experience,
            projects=projects,
            strengths=strengths[:5],
            skill_gaps=gaps[:5],
            suitable_roles=likely_roles[:5],
            resume_quality_score=score,
            resume_quality_assessment=f"Local evidence-based analysis produced a resume quality score of {score}/100. Gemini was unavailable or quota-limited, so the result is based on extractable resume content rather than generated claims.",
            recommended_interview_topics=topics or ["Core programming", "Problem solving", "Projects"],
            recommendations=recommendations,
        )

    def analyze_resume(
        self,
        resume_text: str,
    ) -> Optional[ResumeAnalysisResult]:

        """
        Analyze extracted resume text using Gemini.

        The resume text should already have been extracted from a PDF
        by the API layer.
        """

        resume_text = _clean_text(resume_text)

        if not resume_text:
            logger.warning("Resume analysis requested with empty resume text.")
            return None

        if not self.configured:
            logger.warning("Gemini unavailable. Using local resume intelligence.")
            return self._local_resume_analysis(resume_text)

        # Keep the AI input bounded.
        resume_text = resume_text[:30000]

        prompt = f"""
Analyze the following candidate resume.

================ RESUME TEXT ================

{resume_text}

================ END RESUME =================

Produce a structured resume analysis.

Before producing the result, carefully identify:

- programming languages
- frameworks and libraries
- data analytics skills
- databases
- web development technologies
- developer/data tools
- internships and experience
- education
- projects
- actual strengths
- realistic skill gaps
- suitable entry-level roles
- recommended interview topics

Important:

- Do not invent information.
- Do not upgrade beginner/basic knowledge into advanced expertise.
- Do not add technologies that are not present.
- Do not confuse project names with technologies.
- Do not treat the resume's website/footer text as a candidate skill.
- Ignore document-generation artifacts such as:
  "Powered by", website footers, page numbers, formatting artifacts,
  broken characters, and PDF extraction noise.
- If a category has no supported information, return an empty list.
- Keep recommendations specific to the resume.

Return ONLY the structured response requested by the schema.
""".strip()

        result = self._generate_structured(
            system_instruction=RESUME_ANALYSIS_SYSTEM_PROMPT,
            prompt=prompt,
            response_schema=ResumeAnalysisResult,
        )

        if result is None:
            logger.warning("Gemini resume analysis failed. Using local resume intelligence.")
            return self._local_resume_analysis(resume_text)

        return result


    # ----------------------------------------------------------------
    # Job Description Intelligence
    # ----------------------------------------------------------------

    @staticmethod
    def _local_job_description_analysis(jd_text: str) -> JDAnalysisResult:
        """Deterministic JD intelligence parser for offline/quota-limited demos."""
        text = _clean_text(jd_text)
        lower = text.lower()

        catalog = {
            "programming_languages": [
                "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "SQL", "R", "Go", "PHP"
            ],
            "frameworks_and_libraries": [
                "React", "Node.js", "Express", "FastAPI", "Django", "Flask", "Angular", "Vue.js", "Next.js",
                "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "OpenCV", "Bootstrap", "Tailwind CSS"
            ],
            "databases": [
                "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "SQL Server"
            ],
            "tools_and_platforms": [
                "Git", "GitHub", "Docker", "AWS", "Azure", "GCP", "Linux", "Jupyter", "Postman",
                "Power BI", "Tableau", "Jira", "CI/CD"
            ],
            "soft_skills": [
                "communication", "teamwork", "collaboration", "problem solving", "leadership",
                "adaptability", "time management", "analytical thinking", "attention to detail"
            ],
        }

        aliases = {
            "Node.js": ["node.js", "nodejs", "node"],
            "C++": ["c++"],
            "SQL Server": ["sql server", "mssql"],
            "scikit-learn": ["scikit-learn", "sklearn"],
            "Power BI": ["power bi", "powerbi"],
            "Tailwind CSS": ["tailwind css", "tailwind"],
            "CI/CD": ["ci/cd", "continuous integration", "continuous delivery", "continuous deployment"],
        }

        def present(item: str) -> bool:
            return any(alias in lower for alias in aliases.get(item, [item.lower()]))

        def canonical(item: str) -> str:
            return item

        out = {
            key: [canonical(item) for item in values if present(item)]
            for key, values in catalog.items()
        }

        all_tech = (
            out["programming_languages"]
            + out["frameworks_and_libraries"]
            + out["databases"]
            + out["tools_and_platforms"]
        )

        # Required/preferred classification based on the nearby wording.
        required: List[str] = []
        preferred: List[str] = []
        for tech in all_tech:
            aliases_for_tech = aliases.get(tech, [tech.lower()])
            positions = [lower.find(a) for a in aliases_for_tech if lower.find(a) >= 0]
            pos = min(positions) if positions else -1
            window = lower[max(0, pos - 120): pos + len(tech) + 120] if pos >= 0 else ""
            if any(w in window for w in ["preferred", "nice to have", "nice-to-have", "plus", "bonus", "desired", "optional"]):
                preferred.append(tech)
            else:
                required.append(tech)

        # Role extraction: explicit title first, then common role phrases.
        role = ""
        role_patterns = [
            r"(?:job\s+title|position|role)\s*[:\-]?\s*([^\n|]{2,80})",
            r"(?:hiring|looking\s+for|seeking)\s+(?:a|an)?\s*([A-Za-z][A-Za-z0-9 /&+()._-]{2,70})",
        ]
        for pattern in role_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .:-")
                candidate = re.split(r"\b(?:with|who|to|for)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                if 2 <= len(candidate) <= 80:
                    role = candidate
                    break

        if not role:
            role_candidates = [
                "Full Stack Python Developer", "Full Stack Developer", "Backend Python Developer",
                "Python Developer", "Backend Developer", "Frontend Developer", "Software Engineer",
                "AI/ML Engineer", "Machine Learning Engineer", "Data Scientist", "Data Analyst"
            ]
            role = next((candidate for candidate in role_candidates if candidate.lower() in lower), "Software Engineer")

        lines = [
            re.sub(r"\s+", " ", line).strip(" -•|\t")
            for line in text.splitlines()
            if line.strip()
        ]

        def is_heading(line: str, names: List[str]) -> bool:
            normalized = line.lower().strip().rstrip(":").strip()
            return any(normalized == name or normalized.startswith(name + " ") for name in names)

        def section_items(names: List[str], stop_names: Optional[List[str]] = None, limit: int = 10) -> List[str]:
            values: List[str] = []
            active = False
            stop_names = stop_names or [
                "requirements", "qualifications", "responsibilities", "skills", "education",
                "experience", "preferred qualifications", "preferred skills", "nice to have"
            ]
            for line in lines:
                if is_heading(line, names):
                    active = True
                    continue
                if active and is_heading(line, stop_names) and not is_heading(line, names):
                    break
                if active and len(line) >= 12:
                    values.append(line)
                if len(values) >= limit:
                    break
            return values

        responsibilities = section_items(
            ["responsibilities", "what you will do", "what you'll do", "key responsibilities", "duties"]
        )
        requirements_text = section_items(
            ["requirements", "qualifications", "required qualifications", "required skills", "skills required"]
        )
        education = section_items(["education", "degree", "educational qualifications", "qualification"], limit=6)
        experience = section_items(["experience", "work experience", "professional experience", "experience required"], limit=8)
        preferred_text = section_items(["preferred qualifications", "preferred skills", "nice to have", "preferred"], limit=8)

        # Fallback extraction for paragraph-style JDs without headings.
        if not responsibilities:
            responsibilities = [
                line for line in lines
                if re.search(r"\b(develop|build|design|implement|maintain|analy[sz]e|deploy|integrate|debug|test|create|collaborate|optimize)\b", line, re.IGNORECASE)
            ][:10]

        if not experience:
            experience = [
                line for line in lines
                if re.search(r"\b\d+\+?\s*(?:years?|yrs?)\b", line, re.IGNORECASE)
            ][:8]

        if not education:
            education = [
                line for line in lines
                if re.search(r"\b(bachelor|master|b\.tech|btech|m\.tech|mtech|degree|computer science|engineering)\b", line, re.IGNORECASE)
            ][:6]

        # Explicit experience/education extraction makes sparse JDs more useful.
        experience_matches = re.findall(r"\b\d+\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|professional experience)?\b", text, re.IGNORECASE)
        if experience_matches and not any(experience_matches[0].lower() in item.lower() for item in experience):
            experience.insert(0, experience_matches[0].strip())

        # Interview topics: technology + core role competencies.
        topics: List[str] = []
        topic_defaults = [
            "Python fundamentals", "OOP", "REST API development", "Database and SQL concepts",
            "Debugging and testing", "Git and version control", "Project discussion",
            "Problem solving", "System design basics", "Behavioral scenarios"
        ]
        for item in required + preferred + topic_defaults:
            if item not in topics:
                topics.append(item)

        # Soft skills should be presented in title case for the UI.
        soft_skills = []
        for item in out["soft_skills"]:
            display = item.title()
            if display not in soft_skills:
                soft_skills.append(display)

        # More informative summary from the first meaningful paragraph/line.
        summary_candidates = [
            line for line in lines
            if len(line) >= 40
            and not re.match(r"^(requirements?|qualifications?|responsibilities|skills?|education|experience|preferred)\s*:??$", line, re.IGNORECASE)
        ]
        summary = summary_candidates[0] if summary_candidates else (
            f"{role} opportunity requiring {', '.join(required[:6]) or 'relevant technical skills'}."
        )

        return JDAnalysisResult(
            job_role=role,
            job_summary=summary,
            required_skills=list(dict.fromkeys(required))[:20],
            preferred_skills=list(dict.fromkeys(preferred))[:15],
            programming_languages=out["programming_languages"],
            frameworks_and_libraries=out["frameworks_and_libraries"],
            databases=out["databases"],
            tools_and_platforms=out["tools_and_platforms"],
            responsibilities=responsibilities[:10],
            soft_skills=soft_skills[:10],
            experience_requirements=experience[:8],
            education_requirements=education[:6],
            important_interview_topics=topics[:15],
        )

    @staticmethod
    def _model_to_dict(value: Any) -> Dict[str, Any]:
        """Convert a Pydantic/model-like result into a plain dictionary."""
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return {}

    @staticmethod
    def _normalize_job_analysis_result(
        jd_text: str,
        result: Any,
    ) -> JDAnalysisResult:
        """Normalize Gemini JD output so every concrete technology in the JD is
        represented consistently across required/preferred and category fields.

        Gemini remains useful for semantic extraction, but deterministic
        technology detection prevents omissions such as returning only
        'REST APIs' for a JD that explicitly requires Python, FastAPI/Django,
        MySQL, Docker and AWS as well.
        """
        data = LLMService._model_to_dict(result)
        text = _clean_text(jd_text)
        low = text.lower()

        catalog = {
            "programming_languages": [
                "Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript",
                "SQL", "R", "Go", "PHP",
            ],
            "frameworks_and_libraries": [
                "React", "Node.js", "Express.js", "FastAPI", "Django", "Flask",
                "Angular", "Vue.js", "Next.js", "TensorFlow", "PyTorch",
                "scikit-learn", "Pandas", "NumPy", "OpenCV", "Bootstrap", "Tailwind CSS",
            ],
            "databases": [
                "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "SQL Server",
            ],
            "tools_and_platforms": [
                "Git", "GitHub", "Docker", "AWS", "Azure", "GCP", "Linux", "Jupyter",
                "Postman", "Power BI", "Tableau", "Jira", "CI/CD",
            ],
        }

        aliases = {
            "rest api": "REST APIs", "rest apis": "REST APIs", "restful api": "REST APIs",
            "restful apis": "REST APIs", "nodejs": "Node.js", "express": "Express.js",
            "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "amazon web services": "AWS",
            "google cloud platform": "GCP", "microsoft azure": "Azure",
        }

        def present(term: str) -> bool:
            pattern = re.escape(term.lower())
            if term.lower() in {"c", "r"}:
                return bool(re.search(rf"(?<![a-z]){pattern}(?![a-z])", low))
            return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", low))

        detected: Dict[str, List[str]] = {key: [] for key in catalog}
        for category, terms in catalog.items():
            for term in terms:
                if present(term):
                    detected[category].append(term)

        if re.search(r"\brest(?:ful)?\s+apis?\b", low):
            detected["tools_and_platforms"].append("REST APIs")

        for category, values in detected.items():
            merged = []
            for value in list(data.get(category, []) or []) + values:
                value = _clean_text(value)
                if not value:
                    continue
                if not any(value.lower() == existing.lower() for existing in merged):
                    merged.append(value)
            data[category] = merged

        # Every concrete technology detected in the JD is a requirement unless
        # the wording around it clearly marks it as preferred/optional.
        tech_values = []
        for category_values in detected.values():
            tech_values.extend(category_values)

        required = [_clean_text(x) for x in (data.get("required_skills", []) or []) if _clean_text(x)]
        preferred = [_clean_text(x) for x in (data.get("preferred_skills", []) or []) if _clean_text(x)]

        def nearby_is_preferred(term: str) -> bool:
            for match in re.finditer(re.escape(term.lower()), low):
                window = low[max(0, match.start() - 100): min(len(low), match.end() + 100)]
                if re.search(r"\b(preferred|preferable|nice to have|bonus|plus|optional|desired)\b", window):
                    return True
            return False

        def add_unique(target: List[str], value: str):
            if not any(value.lower() == existing.lower() for existing in target):
                target.append(value)

        for term in tech_values:
            if nearby_is_preferred(term):
                add_unique(preferred, term)
            else:
                add_unique(required, term)
                # If Gemini placed an explicitly required technology in the
                # preferred list, remove that duplicate from preferred.
                preferred = [x for x in preferred if x.lower() != term.lower()]

        data["required_skills"] = required
        data["preferred_skills"] = preferred

        topics = [_clean_text(x) for x in (data.get("important_interview_topics", []) or []) if _clean_text(x)]
        for term in tech_values:
            if not any(term.lower() in topic.lower() or topic.lower() in term.lower() for topic in topics):
                topics.append(term)
        defaults = [
            "Technical fundamentals for the target role",
            "Project and practical implementation questions",
            "Problem solving and debugging",
        ]
        for topic in defaults:
            if topic not in topics:
                topics.append(topic)
        data["important_interview_topics"] = topics[:15]

        # Validate through the existing schema so downstream code receives the
        # same type regardless of whether Gemini or local analysis was used.
        try:
            return JDAnalysisResult(**data)
        except Exception:
            logger.exception("Could not validate normalized JD analysis; using local parser.")
            return LLMService._local_job_description_analysis(text)

    @staticmethod
    def _authoritative_match_with_gemini_narrative(
        resume_analysis: Dict[str, Any],
        jd_analysis: Dict[str, Any],
        gemini_result: Any,
    ) -> JDMatchResult:
        """Use deterministic requirement coverage for lists/score while
        retaining Gemini's narrative when available.

        This prevents resume-only skills (e.g. GitHub or SQLAlchemy) from being
        incorrectly displayed as matched JD requirements.
        """
        local = LLMService._local_resume_job_match(resume_analysis, jd_analysis)
        gemini = LLMService._model_to_dict(gemini_result)

        assessment = _clean_text(gemini.get("match_assessment")) or local.match_assessment
        preparation = gemini.get("preparation_recommendations") or local.preparation_recommendations
        if not isinstance(preparation, list):
            preparation = local.preparation_recommendations

        return JDMatchResult(
            match_score=local.match_score,
            match_assessment=assessment,
            matched_skills=local.matched_skills,
            partially_matched_skills=local.partially_matched_skills,
            missing_skills=local.missing_skills,
            additional_candidate_skills=local.additional_candidate_skills,
            matched_requirements=local.matched_requirements,
            unmet_requirements=local.unmet_requirements,
            recommended_interview_topics=local.recommended_interview_topics,
            preparation_recommendations=preparation,
        )

    def analyze_job_description(
        self,
        jd_text: str,
    ) -> Optional[JDAnalysisResult]:

        jd_text = _clean_text(jd_text)

        if not jd_text:
            logger.warning("JD analysis requested with empty text.")
            return None

        if not self.configured:
            logger.warning("Gemini unavailable. Using local JD intelligence.")
            return self._local_job_description_analysis(jd_text)

        jd_text = jd_text[:30000]

        prompt = f"""
Analyze the following job description.

================ JOB DESCRIPTION ================

{jd_text}

================ END JOB DESCRIPTION ============

Extract:

- target job role
- job summary
- required skills
- preferred skills
- programming languages
- frameworks and libraries
- databases
- tools and platforms
- responsibilities
- soft skills
- experience requirements
- education requirements
- important interview topics

Important:

- Do not invent information.
- Separate required and preferred skills.
- IMPORTANT: every concrete technology explicitly requested by the JD must be captured.
  For example, if the JD says "Python, FastAPI/Django, REST APIs, MySQL, Docker, and AWS",
  do NOT return only REST APIs. Capture Python, FastAPI, Django, REST APIs, MySQL, Docker, and AWS
  in the appropriate category fields and include required technologies in required_skills unless the JD
  clearly marks them as preferred/optional.
- Treat slash-separated alternatives such as "FastAPI/Django" as two technologies.
- Preserve the meaning of the job description.
- Return empty lists when information is unavailable.
- Return ONLY the structured response requested by the schema.
""".strip()

        result = self._generate_structured(
            system_instruction=JD_ANALYSIS_SYSTEM_PROMPT,
            prompt=prompt,
            response_schema=JDAnalysisResult,
        )

        if result is None:
            logger.warning("Gemini JD analysis failed. Using local JD intelligence.")
            return self._local_job_description_analysis(jd_text)

        return self._normalize_job_analysis_result(jd_text, result)

    # ----------------------------------------------------------------
    # Resume ↔ Job Description Matching
    # ----------------------------------------------------------------

    @staticmethod
    def _local_resume_job_match(
        resume_analysis: Dict[str, Any],
        jd_analysis: Dict[str, Any],
    ) -> JDMatchResult:
        """Deterministic Resume ↔ JD matcher used when Gemini is unavailable."""

        def normalize(value: Any) -> str:
            value = _clean_text(value).lower()
            value = re.sub(r"[^a-z0-9+#./\- ]+", " ", value)
            return re.sub(r"\s+", " ", value).strip()

        def collect(data: Dict[str, Any], keys: List[str]) -> List[str]:
            values: List[str] = []
            for key in keys:
                raw = data.get(key, [])
                if isinstance(raw, list):
                    values.extend(_clean_text(item) for item in raw if _clean_text(item))
                elif raw:
                    values.append(_clean_text(raw))
            return values

        resume_items = collect(resume_analysis, [
            "technical_skills",
            "data_analytics_skills",
            "web_development_skills",
            "tools_and_technologies",
        ])
        required_items = collect(jd_analysis, [
            "required_skills",
            "programming_languages",
            "frameworks_and_libraries",
            "databases",
            "tools_and_platforms",
        ])
        preferred_items = collect(jd_analysis, ["preferred_skills"])

        def unique_map(items: List[str]) -> Dict[str, str]:
            result: Dict[str, str] = {}
            for item in items:
                key = normalize(item)
                if key:
                    result.setdefault(key, item)
            return result

        resume_map = unique_map(resume_items)
        required_map = unique_map(required_items)
        preferred_map = unique_map(preferred_items)

        matched: List[str] = []
        partial: List[str] = []
        missing: List[str] = []
        used_resume: set[str] = set()

        for jd_key, jd_value in required_map.items():
            if jd_key in resume_map:
                matched.append(jd_value)
                used_resume.add(jd_key)
                continue

            jd_tokens = {x for x in re.findall(r"[a-z0-9+#.]+", jd_key) if len(x) >= 3}
            best_key = None
            best_overlap = 0.0

            for resume_key in resume_map:
                if resume_key in used_resume:
                    continue
                resume_tokens = {
                    x for x in re.findall(r"[a-z0-9+#.]+", resume_key) if len(x) >= 3
                }
                if not jd_tokens or not resume_tokens:
                    continue
                overlap = len(jd_tokens & resume_tokens) / len(jd_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_key = resume_key

            if best_key is not None and best_overlap >= 0.5:
                partial.append(jd_value)
                used_resume.add(best_key)
            else:
                missing.append(jd_value)

        preferred_matched = [
            value for key, value in preferred_map.items() if key in resume_map
        ]

        required_total = len(required_map)
        required_score = (
            ((len(matched) + 0.5 * len(partial)) / required_total) * 80
            if required_total else 80
        )

        preferred_total = len(preferred_map)
        preferred_score = (
            (len(preferred_matched) / preferred_total) * 20
            if preferred_total else 20
        )

        match_score = round(max(0, min(100, required_score + preferred_score)))

        job_keys = set(required_map) | set(preferred_map)
        additional = [
            value for key, value in resume_map.items() if key not in job_keys
        ]

        matched_requirements = [f"Demonstrated skill: {skill}" for skill in matched]
        unmet_requirements = [
            f"Resume evidence not found for: {skill}" for skill in missing
        ]

        topics = collect(jd_analysis, ["important_interview_topics"])
        recommended_topics = list(topics)

        for skill in reversed(missing + partial):
            if skill in recommended_topics:
                recommended_topics.remove(skill)
            recommended_topics.insert(0, skill)

        recommended_topics = recommended_topics[:10]

        preparation = [
            f"Practice {skill} with a small hands-on problem."
            for skill in missing
        ]
        preparation += [
            f"Strengthen practical interview-ready knowledge of {skill}."
            for skill in partial
        ]
        if not preparation:
            preparation.append(
                "Prepare project-based examples demonstrating the major job requirements."
            )

        if match_score >= 85:
            assessment = (
                "The resume shows strong alignment with the analyzed job description. "
                "Most required skills have direct resume evidence."
            )
        elif match_score >= 70:
            assessment = (
                "The resume shows good alignment with the analyzed job description, "
                "with some areas needing stronger evidence or preparation."
            )
        elif match_score >= 50:
            assessment = (
                "The resume shows partial alignment with the analyzed job description. "
                "Several requirements need additional preparation or evidence."
            )
        else:
            assessment = (
                "The resume shows limited alignment with the analyzed job description. "
                "Several required skills are not represented in the analyzed resume."
            )

        return JDMatchResult(
            match_score=match_score,
            match_assessment=assessment,
            matched_skills=matched,
            partially_matched_skills=partial,
            missing_skills=missing,
            additional_candidate_skills=additional,
            matched_requirements=matched_requirements,
            unmet_requirements=unmet_requirements,
            recommended_interview_topics=recommended_topics,
            preparation_recommendations=preparation,
        )

    def match_resume_to_job(
        self,
        resume_analysis: Dict[str, Any],
        jd_analysis: Dict[str, Any],
    ) -> Optional[JDMatchResult]:
        """Compare the analyzed resume with an analyzed job description."""

        if not self.configured:
            return self._local_resume_job_match(resume_analysis, jd_analysis)

        # Keep the matching payload focused on evidence that matters for
        # recruitment. This also avoids unnecessarily large Gemini prompts.
        resume_profile = {
            "candidate_summary": resume_analysis.get("candidate_summary", ""),
            "technical_skills": resume_analysis.get("technical_skills", []),
            "data_analytics_skills": resume_analysis.get("data_analytics_skills", []),
            "web_development_skills": resume_analysis.get("web_development_skills", []),
            "tools_and_technologies": resume_analysis.get("tools_and_technologies", []),
            "education": resume_analysis.get("education", []),
            "experience": resume_analysis.get("experience", []),
            "projects": resume_analysis.get("projects", []),
            "strengths": resume_analysis.get("strengths", []),
            "skill_gaps": resume_analysis.get("skill_gaps", []),
            "suitable_roles": resume_analysis.get("suitable_roles", []),
        }

        job_profile = {
            "job_role": jd_analysis.get("job_role", ""),
            "job_summary": jd_analysis.get("job_summary", ""),
            "required_skills": jd_analysis.get("required_skills", []),
            "preferred_skills": jd_analysis.get("preferred_skills", []),
            "programming_languages": jd_analysis.get("programming_languages", []),
            "frameworks_and_libraries": jd_analysis.get("frameworks_and_libraries", []),
            "databases": jd_analysis.get("databases", []),
            "tools_and_platforms": jd_analysis.get("tools_and_platforms", []),
            "responsibilities": jd_analysis.get("responsibilities", []),
            "soft_skills": jd_analysis.get("soft_skills", []),
            "experience_requirements": jd_analysis.get("experience_requirements", []),
            "education_requirements": jd_analysis.get("education_requirements", []),
            "important_interview_topics": jd_analysis.get("important_interview_topics", []),
        }

        prompt = f"""
Compare the candidate resume profile with the job description.

================ RESUME PROFILE ================
{json.dumps(resume_profile, ensure_ascii=False)}

================ JOB DESCRIPTION ===============
{json.dumps(job_profile, ensure_ascii=False)}
=================================================

Calculate a realistic resume-to-job match.

Rules for the match score:
- Return an integer from 0 to 100.
- Required skills and responsibilities matter more than preferred skills.
- A skill counts as matched only when the resume provides evidence for it.
- A skill can be partially matched when the candidate has related evidence but not the exact requested skill.
- Missing resume evidence is a preparation gap, not proof that the candidate cannot perform the skill.
- Do not invent candidate skills, experience, projects, or achievements.
- Do not make a hiring decision.

Return EXACTLY these fields:
- match_score
- match_assessment
- matched_skills
- partially_matched_skills
- missing_skills
- additional_candidate_skills
- matched_requirements
- unmet_requirements
- recommended_interview_topics
- preparation_recommendations

Return ONLY valid JSON.
""".strip()

        # First try Gemini's native structured output.
        result = self._generate_structured(
            system_instruction=JD_MATCHING_SYSTEM_PROMPT,
            prompt=prompt,
            response_schema=JDMatchResult,
        )

        if result is not None:
            return self._authoritative_match_with_gemini_narrative(
                resume_analysis, jd_analysis, result
            )

        # Some Gemini/model combinations can reject a complex Pydantic schema
        # even though the same model handles normal JSON successfully. Retry
        # without response_schema, then validate the JSON ourselves.
        logger.warning(
            "Retrying resume-JD matching with schema-free JSON generation."
        )

        result = self._generate_json_without_schema(
            system_instruction=JD_MATCHING_SYSTEM_PROMPT,
            prompt=prompt,
            response_schema=JDMatchResult,
        )

        if result is not None:
            return self._authoritative_match_with_gemini_narrative(
                resume_analysis, jd_analysis, result
            )

        logger.warning(
            "Gemini resume-JD matching failed. Using local deterministic matcher."
        )

        return self._local_resume_job_match(
            resume_analysis=resume_analysis,
            jd_analysis=jd_analysis,
        )



    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------


    @staticmethod
    def _local_evaluate(
        question: str,
        answer: str,
        role: str,
        experience: str,
        difficulty: str,
        interview_type: str,
    ) -> EvaluationResult:
        """Deterministic evaluator used when Gemini is unavailable/quota-limited.

        This is intentionally conservative: it rewards evidence in the answer
        instead of assuming that an answer is correct just because it is long.
        """
        q = _clean_text(question)
        a = _clean_text(answer)
        bank_item = find_question(q) or find_similar_question(q)
        bank_concepts = set()
        if bank_item:
            for concept in bank_item.get("expected_concepts", []):
                normalized = re.sub(r"[^a-z0-9\s+#.-]", " ", str(concept).lower())
                bank_concepts.add(re.sub(r"\s+", " ", normalized).strip())

        if not a:
            return EvaluationResult(
                correctness=0,
                technical_knowledge=0,
                relevance=0,
                clarity=0,
                communication=0,
                overall_score=0,
                strengths=[],
                weaknesses=["No meaningful answer was provided."],
                missing_points=["Provide a direct answer to the question."],
                improvement_tips=[
                    "Answer the question directly.",
                    "Include the relevant technical concept.",
                    "Add a short example or reasoning when appropriate.",
                ],
                ideal_answer="Provide a concise, technically relevant answer.",
                source="local",
                fallback_reason="Gemini unavailable; local evaluator used.",
            )

        normalized_answer = re.sub(r"[^a-z0-9+#./_\s-]", " ", a.lower())
        answer_words = re.findall(r"[a-z0-9+#./_-]+", normalized_answer)
        answer_set = set(answer_words)
        word_count = len(answer_words)

        # Extract useful technical tokens from the question. Stop words are
        # excluded so generic words do not inflate the score.
        stop_words = {
            "what", "are", "the", "most", "important", "technical", "skills",
            "for", "a", "an", "and", "or", "of", "to", "in", "on", "is",
            "this", "that", "how", "would", "you", "your", "explain",
            "one", "problem", "expect", "role", "solve", "debug", "program",
            "produces", "unexpected", "result", "improve", "performance",
            "slow", "application", "concept", "from", "field", "using",
            "simple", "example", "trade", "offs", "between", "two",
            "approaches", "test", "feature", "before", "considering",
            "complete", "describe", "structured", "difficult",
        }
        question_tokens = [
            token for token in re.findall(r"[a-z0-9+#.]+", q.lower())
            if len(token) >= 3 and token not in stop_words
        ]

        # Common technical concepts associated with generic software-engineering
        # questions. These are evidence checks, not claims that the candidate
        # knows a concept.
        concept_groups = {
            "debug": {
                "debug", "debugger", "logging", "logs", "breakpoint", "trace",
                "stack", "traceback", "reproduce", "isolate", "test", "input",
                "error", "exception",
            },
            "performance": {
                "performance", "profiling", "profile", "algorithm", "complexity",
                "cache", "caching", "database", "query", "index", "memory",
                "cpu", "optimize", "optimization", "benchmark",
            },
            "problem_solving": {
                "requirements", "reproduce", "analyze", "isolate", "hypothesis",
                "test", "solution", "tradeoff", "trade-offs", "implement",
                "verify", "validate",
            },
            "software_engineering": {
                "testing", "unit", "integration", "git", "version", "control",
                "api", "database", "architecture", "design", "deployment",
                "code", "review", "documentation",
            },
        }

        matched_concepts = set()
        for name, terms in concept_groups.items():
            if answer_set.intersection(terms):
                matched_concepts.add(name)

        # Role-specific evidence vocabulary.
        role_terms = set(
            re.findall(r"[a-z0-9+#.]+", _clean_text(role).lower())
        )
        role_evidence = len(role_terms.intersection(answer_set))

        # Metadata-aware concept coverage for curated fallback questions.
        # This is stronger than generic keyword overlap because every bank
        # question declares the concepts a good answer should address.
        concept_hits = 0
        answer_lower = a.lower()
        for concept in bank_concepts:
            aliases = [concept]
            if concept == "open closed": aliases += ["open-closed", "open closed principle"]
            if concept == "single responsibility": aliases += ["single responsibility principle", "srp"]
            if concept == "dependency inversion": aliases += ["dependency inversion principle", "dip"]
            if any(alias in answer_lower for alias in aliases):
                concept_hits += 1
        bank_coverage = (concept_hits / len(bank_concepts)) if bank_concepts else 0.0

        # Relevance: direct lexical overlap plus technical evidence.
        q_overlap = len(set(question_tokens).intersection(answer_set))
        lexical_relevance = min(10, 2 + q_overlap * 2)
        technical_evidence = min(10, len(matched_concepts) * 2 + role_evidence)
        relevance = max(1, min(10, round((lexical_relevance + technical_evidence + bank_coverage * 10) / 3)))

        # Optional TF-IDF similarity gives the offline evaluator an explicit
        # NLP signal without pretending it is equivalent to an LLM judge.
        semantic_similarity = 0.0
        if TfidfVectorizer is not None and cosine_similarity is not None:
            reference = " ".join(sorted(bank_concepts)) or " ".join(question_tokens)
            try:
                matrix = TfidfVectorizer(stop_words="english").fit_transform([reference, a])
                semantic_similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
            except Exception:
                semantic_similarity = 0.0

        # Completeness/technical correctness proxy. We deliberately cap this
        # because a local evaluator cannot truly prove correctness.
        if word_count < 5:
            correctness = 2
        elif word_count < 12:
            correctness = 4
        elif word_count < 25:
            correctness = 6
        elif word_count < 50:
            correctness = 7
        else:
            correctness = 8

        if matched_concepts:
            correctness = min(10, correctness + min(2, len(matched_concepts)))
        if bank_concepts:
            if bank_coverage < 0.20:
                correctness = min(correctness, 3)
            elif bank_coverage < 0.40:
                correctness = min(correctness, 5)
            elif bank_coverage >= 0.60:
                correctness = min(10, correctness + 1)

        technical_knowledge = min(
            10,
            max(1, round(2 + len(matched_concepts) * 2 + min(3, q_overlap) + bank_coverage * 3)),
        )

        # Clarity/communication use conservative, observable text signals.
        sentence_count = max(1, len(re.findall(r"[.!?]+", a)))
        avg_sentence_words = word_count / sentence_count
        clarity = 7
        if word_count < 8:
            clarity = 3
        elif avg_sentence_words > 45:
            clarity = 5
        elif avg_sentence_words > 30:
            clarity = 6

        communication = min(10, max(1, 4 + (1 if word_count >= 12 else 0) +
                                      (1 if word_count >= 25 else 0) +
                                      (1 if sentence_count >= 2 else 0)))

        # Overall score weights technical substance more heavily than prose.
        overall = round(
            correctness * 0.25
            + technical_knowledge * 0.30
            + relevance * 0.20
            + (semantic_similarity * 10) * 0.05
            + clarity * 0.10
            + communication * 0.10
        )
        overall = max(1, min(10, overall))

        strengths = []
        if relevance >= 7:
            strengths.append("The answer addresses the question with relevant points.")
        if technical_knowledge >= 7:
            strengths.append("The answer contains several relevant technical concepts.")
        if communication >= 7:
            strengths.append("The answer is reasonably structured and understandable.")
        if word_count >= 25:
            strengths.append("The answer provides enough detail to support evaluation.")

        weaknesses = []
        missing_points = []
        if relevance < 6:
            weaknesses.append("The answer needs a more direct connection to the question.")
        if technical_knowledge < 6:
            weaknesses.append("The answer contains limited technical evidence.")
        if word_count < 12:
            weaknesses.append("The answer is too brief to demonstrate sufficient depth.")
        if not matched_concepts and not bank_concepts:
            weaknesses.append("Add concrete technical concepts, steps, or examples.")
        if bank_concepts and bank_coverage < 0.5:
            weaknesses.append(f"Address more of the expected concepts for {bank_item.get('topic', 'this topic')}.")
        if not weaknesses:
            weaknesses.append("Add a concrete example or trade-off to make the answer stronger.")

        if "debug" in q.lower() and not matched_concepts.intersection({"debug", "problem_solving"}):
            missing_points.append("Mention a reproducible debugging process such as isolating the issue and testing the suspected cause.")
        elif any(x in q.lower() for x in ("performance", "slow application")) and "performance" not in matched_concepts:
            missing_points.append("Mention profiling, identifying the bottleneck, and validating an optimization.")
        elif "technical skills" in q.lower() and word_count < 20:
            missing_points.append("Name specific technical skills and briefly explain where you have used them.")
        elif bank_concepts and bank_coverage < 0.75:
            missing = [c for c in sorted(bank_concepts) if c not in answer_lower]
            missing_points.append("Cover missing concepts such as " + ", ".join(missing[:3]) + ".")
        else:
            missing_points.append("Support the answer with a concrete technical example or reasoning.")

        improvement_tips = [
            "Answer the exact question first.",
            "Use specific technical terms and explain why they matter.",
            "Add a short practical example when possible.",
        ]

        ideal_answer = (
            "Give a direct answer, explain the relevant technical concept, "
            "describe the main steps or trade-offs, and include a practical example."
        )

        return EvaluationResult(
            correctness=correctness,
            technical_knowledge=technical_knowledge,
            relevance=relevance,
            clarity=clarity,
            communication=communication,
            overall_score=overall,
            strengths=strengths[:4],
            weaknesses=weaknesses[:4],
            missing_points=missing_points[:3],
            improvement_tips=improvement_tips,
            ideal_answer=ideal_answer,
            source="local",
            fallback_reason=(
                "Gemini unavailable or quota-limited; local NLP evaluator used"
                + (" with curated question-bank concepts." if bank_item else ".")
            ),
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        role: str,
        experience: str,
        difficulty: str,
        interview_type: str,
    ) -> EvaluationResult:

        if not self.configured:
            return self._local_evaluate(
                question=question,
                answer=answer,
                role=role,
                experience=experience,
                difficulty=difficulty,
                interview_type=interview_type,
            )

        prompt = evaluation_prompt(
            question=question,
            answer=answer,
            role=role,
            experience=experience,
            difficulty=difficulty,
            interview_type=interview_type,
        )

        result = self._generate_structured(
            system_instruction=EVALUATION_SYSTEM_PROMPT,
            prompt=prompt,
            response_schema=EvaluationResult,
        )

        if result is None:
            return self._local_evaluate(
                question=question,
                answer=answer,
                role=role,
                experience=experience,
                difficulty=difficulty,
                interview_type=interview_type,
            )

        return result

    # ----------------------------------------------------------------
    # Adaptive question generation
    # ----------------------------------------------------------------

    @staticmethod
    def _normalize_question(
        question: str,
    ) -> str:

        text = question.lower().strip()

        text = re.sub(
            r"`[^`]*`",
            "",
            text,
        )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _is_duplicate_question(
        cls,
        candidate: str,
        previous_questions: List[str],
    ) -> bool:

        candidate_normalized = cls._normalize_question(
            candidate
        )

        if not candidate_normalized:
            return True

        candidate_words = set(
            candidate_normalized.split()
        )

        for previous in previous_questions:

            previous_normalized = cls._normalize_question(
                previous
            )

            if not previous_normalized:
                continue

            # Exact match
            if candidate_normalized == previous_normalized:
                return True

            previous_words = set(
                previous_normalized.split()
            )

            if not candidate_words or not previous_words:
                continue

            intersection = candidate_words.intersection(
                previous_words
            )

            union = candidate_words.union(
                previous_words
            )

            similarity = (
                len(intersection)
                / max(len(union), 1)
            )

            # High lexical similarity
            if similarity >= 0.75:

                return True

        return False

    @staticmethod
    def _adaptive_question_directive(
        role: str,
        interview_type: str,
        difficulty: str,
        turns: List[Dict[str, Any]],
        jd_analysis: Optional[Dict[str, Any]],
        jd_match: Optional[Dict[str, Any]],
        domain: str = "General Software Engineering",
        interview_source: str = "new",
    ) -> str:
        """Build an explicit, target-driven adaptive interview policy."""
        jd_analysis = jd_analysis or {}
        jd_match = jd_match or {}

        def as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [_clean_text(x) for x in value if _clean_text(x)]
            value = _clean_text(value)
            return [value] if value else []

        def unique(items: List[str]) -> List[str]:
            seen = set()
            out = []
            for item in items:
                key = re.sub(r"\s+", " ", item.lower()).strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(item)
            return out

        def mentioned(skill: str, text: str) -> bool:
            s = _clean_text(skill).lower()
            q = _clean_text(text).lower()
            aliases = {
                "fastapi": ["fastapi"],
                "django": ["django"],
                "rest apis": ["rest api", "restful api", "rest apis"],
                "docker": ["docker", "containerization", "container"],
                "aws": ["aws", "amazon web services"],
                "mysql": ["mysql"],
                "python": ["python"],
            }
            return any(alias in q for alias in aliases.get(s, [s]))

        missing = unique(as_list(jd_match.get("missing_skills")))
        partial = unique(as_list(jd_match.get("partially_matched_skills")))
        matched = unique(as_list(jd_match.get("matched_skills")))
        topics = unique(as_list(jd_analysis.get("important_interview_topics")))
        frameworks = unique(as_list(jd_analysis.get("frameworks_and_libraries")))
        languages = unique(as_list(jd_analysis.get("programming_languages")))
        databases = unique(as_list(jd_analysis.get("databases")))
        tools = unique(as_list(jd_analysis.get("tools_and_platforms")))

        previous_text = "\n".join(
            _clean_text(turn.get("question")) for turn in turns
        )
        uncovered_missing = [x for x in missing if not mentioned(x, previous_text)]
        uncovered_partial = [x for x in partial if not mentioned(x, previous_text)]

        last_score = None
        if turns:
            try:
                last_score = float(turns[-1].get("overall_score"))
            except (TypeError, ValueError):
                last_score = None

        if not turns:
            stage = "FIRST QUESTION: use the highest-priority uncovered JD gap."
        elif last_score is not None and last_score < 5:
            stage = (
                "WEAK PREVIOUS ANSWER: stay on the previous question's concept and "
                "re-test it with simpler wording and a concrete scenario."
            )
        elif last_score is not None and last_score < 8:
            stage = (
                "AVERAGE PREVIOUS ANSWER: deepen the same concept OR move to the "
                "next uncovered JD requirement. Require implementation detail."
            )
        else:
            stage = (
                "STRONG PREVIOUS ANSWER: move to the next uncovered JD requirement "
                "or increase depth; do not repeat the same concept."
            )

        context_policy = (
            "NEW INTERVIEW: use ONLY the selected role, experience, interview type, difficulty, domain and question count. Ignore saved resume/JD context."
            if interview_source == "new"
            else (
                "DOMAIN PRACTICE: the selected domain is the primary scope. Resume/JD context is secondary and must never pull the interview outside the selected domain."
                if interview_source == "domain_practice"
                else "PROFILE/JOB INTERVIEW: use the selected parameters plus the explicitly supplied resume/JD context."
            )
        )

        return f"""
You are an adaptive interviewer for {role}.
Interview type: {interview_type}. Domain: {domain}. Difficulty: {difficulty}.
Interview source: {interview_source}.

{context_policy}

{stage}

MANDATORY PRIORITY QUEUE:
1. Uncovered missing JD requirements: {json.dumps(uncovered_missing, ensure_ascii=False)}
2. Uncovered partial JD requirements: {json.dumps(uncovered_partial, ensure_ascii=False)}
3. JD interview topics: {json.dumps(topics[:10], ensure_ascii=False)}
4. Matched resume skills: {json.dumps(matched, ensure_ascii=False)}
5. JD technologies: {json.dumps(unique(frameworks + languages + databases + tools)[:12], ensure_ascii=False)}

CRITICAL:
- If an uncovered missing JD requirement exists, the next question MUST test the
  first relevant uncovered requirement unless the previous answer was weak and
  needs remediation.
- Do not replace a missing JD requirement with a generic resume-project question.
- Do not ask about a JD requirement already covered by an earlier question unless
  the previous answer was weak or a harder follow-up is explicitly needed.
- For Technical/Mixed, concrete JD gaps have priority over generic Python theory.
- For Coding, convert the highest-priority uncovered requirement into a practical task.
- For SQL, prioritize the JD database and database operations.
- For HR/Behavioral, use resume evidence only when the interview source allows resume context.
- For New Interview, never mention or ask about uploaded resume/JD information.
- For Domain Practice, stay inside the selected domain even if the resume/JD contains unrelated technologies.
- For Coding/SQL, honor both the selected domain and selected interview type.
- Never invent candidate experience.
- Return exactly one question.
""".strip()

    @classmethod
    def _planned_adaptive_question(
        cls,
        role: str,
        interview_type: str,
        turns: List[Dict[str, Any]],
        jd_analysis: Optional[Dict[str, Any]],
        jd_match: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Return a deterministic high-priority question when adaptation must be reliable.

        This planner intentionally runs before Gemini for concrete JD gaps. Gemini can
        still provide questions after the gap queue is covered, but the core final-year
        demo behavior does not depend on model compliance or quota availability.
        """
        jd_analysis = jd_analysis or {}
        jd_match = jd_match or {}

        def as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [_clean_text(x) for x in value if _clean_text(x)]
            value = _clean_text(value)
            return [value] if value else []

        def unique(items: List[str]) -> List[str]:
            seen = set()
            out = []
            for item in items:
                key = re.sub(r"\s+", " ", item.lower()).strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(item)
            return out

        def mentioned(skill: str, text: str) -> bool:
            s = _clean_text(skill).lower()
            q = _clean_text(text).lower()
            aliases = {
                "rest apis": ["rest api", "restful api", "rest apis"],
                "node.js": ["node.js", "nodejs", "node"],
                "express.js": ["express.js", "express"],
                "amazon web services": ["amazon web services", "aws"],
            }
            return any(alias in q for alias in aliases.get(s, [s]))

        previous_questions = [
            _clean_text(turn.get("question"))
            for turn in turns
            if _clean_text(turn.get("question"))
        ]
        previous_text = "\n".join(previous_questions)

        missing = unique(as_list(jd_match.get("missing_skills")))
        partial = unique(as_list(jd_match.get("partially_matched_skills")))
        topics = unique(as_list(jd_analysis.get("important_interview_topics")))
        languages = unique(as_list(jd_analysis.get("programming_languages")))
        frameworks = unique(as_list(jd_analysis.get("frameworks_and_libraries")))
        databases = unique(as_list(jd_analysis.get("databases")))
        tools = unique(as_list(jd_analysis.get("tools_and_platforms")))

        # Coding and SQL interviews must stay in their selected mode.
        # These branches intentionally run before the generic JD-gap queue,
        # so a Coding interview can never accidentally become a normal
        # FastAPI/HR/technical text question.
        normalized_type = (interview_type or "").lower().strip()

        if "coding" in normalized_type or "programming" in normalized_type:
            coding_candidates = [
                "Write a Python function that accepts a list of integers and returns the first pair whose sum equals a given target. Explain your approach and time/space complexity.",
                "Write a Python function that removes duplicate values from a list while preserving the original order. Explain the complexity.",
                "Write a Python function to find the first non-repeating character in a string. Explain the data structure and complexity you would use.",
            ]

            # If a concrete JD technology is relevant to coding, prefer it,
            # but still make the task a genuine coding problem.
            coding_skill = next(
                (
                    skill for skill in missing + partial + frameworks + languages
                    if any(
                        token in _clean_text(skill).lower()
                        for token in ("python", "fastapi", "django", "javascript", "node")
                    )
                ),
                None,
            )

            if coding_skill:
                coding_candidates.insert(
                    0,
                    f"Write a practical Python coding solution for a {role} backend scenario "
                    f"that demonstrates how you would work with {coding_skill}. Include the "
                    "function or endpoint logic, input validation, error handling, and explain "
                    "time and space complexity.",
                )

            for q in coding_candidates:
                if not cls._is_duplicate_question(q, previous_questions):
                    return q

        if "sql" in normalized_type or "database" in normalized_type:
            db = databases[0] if databases else "MySQL"
            sql_candidates = [
                f"Using {db}, write a query to find the top three customers by total purchase amount. "
                "Use GROUP BY and ORDER BY, and explain how you would handle ties.",
                f"Using {db}, write a query that returns customers who placed more than five orders. "
                "Explain the JOIN, GROUP BY, and HAVING clauses.",
                f"Using {db}, design a query for customer search with filtering, sorting, and pagination. "
                "Explain which indexes you would consider.",
            ]
            for q in sql_candidates:
                if not cls._is_duplicate_question(q, previous_questions):
                    return q

        # Weak answer -> remediate the last concept before moving forward.
        if turns:
            last = turns[-1]
            try:
                score = float(last.get("overall_score"))
            except (TypeError, ValueError):
                score = None
            if score is not None and score < 5:
                last_question = _clean_text(last.get("question"))
                if last_question:
                    remedial = (
                        "Let's revisit the previous concept. "
                        f"In simpler terms, {last_question} "
                        "Explain the answer step by step and give one small practical example."
                    )
                    if not cls._is_duplicate_question(remedial, previous_questions):
                        return remedial

        # For a strong/average answer, move through concrete missing requirements.
        if normalized_type not in {"hr", "behavioral", "coding", "programming", "sql", "database"}:
            for skill in missing:
                if mentioned(skill, previous_text):
                    continue
                low = skill.lower()
                if "fastapi" in low:
                    q = (
                        "The target role requires FastAPI, which is not demonstrated on your resume. "
                        "How would you build a REST endpoint in FastAPI with request validation, "
                        "dependency handling, SQLAlchemy/MySQL access, and appropriate HTTP error responses?"
                    )
                elif "django" in low:
                    q = (
                        "The target role requires Django, which is not demonstrated on your resume. "
                        "Explain how you would structure a small Django backend for a REST API, including "
                        "models, views, URL routing, validation, and database access."
                    )
                elif "rest" in low and "api" in low:
                    q = (
                        "REST APIs are a requirement for this role. Design a simple customer API and explain "
                        "the HTTP methods, resource URLs, status codes, request validation, authentication, "
                        "error handling, and JSON response structure you would use."
                    )
                elif "docker" in low or "container" in low:
                    q = (
                        "Docker is a requirement for this role. You have a Python backend and MySQL database. "
                        "Explain how you would containerize the backend, configure its environment variables, "
                        "connect it to MySQL, and run the application consistently across environments."
                    )
                elif low == "aws" or "aws" in low or "amazon web services" in low:
                    q = (
                        "AWS is a requirement for this role. Suppose your Python backend is ready locally. "
                        "Describe a practical deployment approach on AWS, including where you would run the "
                        "application, how you would handle the database, environment secrets, logging, and scaling."
                    )
                else:
                    q = (
                        f"The target role requires {skill}. Explain what {skill} is, how you would use it "
                        "in a Python backend project, and one important implementation or design consideration."
                    )
                if not cls._is_duplicate_question(q, previous_questions):
                    return q

            for skill in partial:
                if mentioned(skill, previous_text):
                    continue
                q = (
                    f"You have related experience but only partial evidence for {skill}. "
                    f"How would you apply {skill} in a {role} project, and what would you need to learn or implement "
                    "to use it confidently in production?"
                )
                if not cls._is_duplicate_question(q, previous_questions):
                    return q

        # Once the gap queue is covered, use JD topics before generic questions.
        for topic in topics:
            q = (
                f"For a {role} position, explain the key concepts behind {topic} "
                "and give a practical example relevant to a backend application."
            )
            if not cls._is_duplicate_question(q, previous_questions):
                return q

        # Database fallback for SQL/database interviews.
        if "sql" in interview_type.lower() or "database" in interview_type.lower():
            db = databases[0] if databases else "MySQL"
            q = (
                f"Using {db}, explain how you would design and optimize a query that retrieves "
                "customer records with filtering, sorting, and pagination. Mention indexing and why it helps."
            )
            if not cls._is_duplicate_question(q, previous_questions):
                return q

        # Keep remaining JD technologies in the queue.
        for skill in unique(frameworks + databases + tools):
            q = (
                f"Explain one important {skill} concept that a {role} candidate should know, "
                "and show how you would apply it in a practical project."
            )
            if not cls._is_duplicate_question(q, previous_questions):
                return q

        return None

    @classmethod
    def _question_bank_fallback(
        cls,
        domain: str,
        difficulty: str,
        interview_type: str,
        previous_questions: List[str],
        turns: List[Dict[str, Any]],
    ) -> Optional[QuestionResult]:
        """Select a unique curated question when Gemini cannot respond.

        Selection is deterministic and adaptive: it prefers the current
        difficulty, avoids previously asked questions, and rotates topics.
        """
        items = questions_for_domain(domain)
        used = {cls._normalize_question(q) for q in previous_questions}
        current = (difficulty or "Medium").title()
        if current not in {"Easy", "Medium", "Hard"}:
            current = "Medium"

        # Prefer exact current difficulty, then nearest difficulty.
        order = {"Easy": ["Easy", "Medium", "Hard"],
                 "Medium": ["Medium", "Easy", "Hard"],
                 "Hard": ["Hard", "Medium", "Easy"]}[current]

        used_topics = [str(t.get("question", "")) for t in turns[-4:]]
        for level in order:
            candidates = [x for x in items if x.get("difficulty") == level and cls._normalize_question(x["question"]) not in used]
            # Rotate to an unused topic before reusing a recent topic.
            for item in candidates:
                if not any(str(item.get("topic", "")).lower() in q.lower() for q in used_topics):
                    return QuestionResult(
                        question=item["question"],
                        source="question-bank",
                        fallback_reason=(
                            "Gemini was unavailable, so a unique curated fallback question "
                            f"for {domain} was selected. Difficulty: {level}."
                        ),
                    )
            if candidates:
                item = candidates[0]
                return QuestionResult(
                    question=item["question"],
                    source="question-bank",
                    fallback_reason="Gemini was unavailable; unique curated question-bank fallback used.",
                )
        return None

    def next_question(
        self,
        role: str,
        experience: str,
        interview_type: str,
        difficulty: str,
        turns: List[Dict[str, Any]],
        resume_analysis: Optional[Dict[str, Any]] = None,
        jd_analysis: Optional[Dict[str, Any]] = None,
        jd_match: Optional[Dict[str, Any]] = None,
        domain: str = "General Software Engineering",
        interview_source: str = "new",
    ) -> QuestionResult:

        previous_questions = [
            _clean_text(turn.get("question"))
            for turn in turns
            if _clean_text(turn.get("question"))
        ]

        # ------------------------------------------------------------
        # Deterministic planner comes FIRST for concrete JD gaps.
        # This guarantees real adaptive behavior even when Gemini is
        # quota-limited or ignores a long natural-language directive.
        # ------------------------------------------------------------
        planned = None
        if interview_source in {"candidate_profile", "job_specific"} and (jd_analysis or jd_match):
            planned = self._planned_adaptive_question(
                role=role,
                interview_type=interview_type,
                turns=turns,
                jd_analysis=jd_analysis,
                jd_match=jd_match,
            )

        if planned:
            return QuestionResult(
                question=planned,
                source="adaptive-planner",
                fallback_reason=(
                    "Deterministic interview planner selected the highest-priority "
                    "JD requirement or answer-driven follow-up."
                ),
            )

        if not self.configured:
            bank_result = self._question_bank_fallback(
                domain=domain,
                difficulty=difficulty,
                interview_type=interview_type,
                previous_questions=previous_questions,
                turns=turns,
            )
            if bank_result is not None:
                bank_result.fallback_reason = "Gemini is not configured; curated offline question bank used."
                return bank_result
            return QuestionResult(
                question=self._personalized_fallback_question(
                    role=role, index=len(turns), resume_analysis=resume_analysis,
                    jd_analysis=jd_analysis, jd_match=jd_match,
                    interview_type=interview_type, previous_questions=previous_questions, turns=turns,
                ),
                source="safe-fallback",
                fallback_reason="Gemini is not configured and the curated bank had no unused question.",
            )

        # Gemini handles the remaining open-ended questions after the
        # deterministic high-priority queue has been covered.
        for attempt in range(3):
            prompt = question_prompt(
                role=role,
                experience=experience,
                interview_type=interview_type,
                difficulty=difficulty,
                turns=turns,
                resume_analysis=resume_analysis,
                jd_analysis=jd_analysis,
                jd_match=jd_match,
                domain=domain,
                interview_source=interview_source,
            )

            adaptive_directive = self._adaptive_question_directive(
                role=role,
                interview_type=interview_type,
                difficulty=difficulty,
                turns=turns,
                jd_analysis=jd_analysis,
                jd_match=jd_match,
            )
            prompt += (
                "\n\n================ ADAPTIVE INTERVIEW POLICY ================\n"
                f"{adaptive_directive}\n"
                "=============================================================="
            )

            if attempt > 0:
                prompt += f"""

IMPORTANT RETRY #{attempt}:
Generate a substantially different question.
Previous questions that MUST be avoided:
{chr(10).join(f'- {q}' for q in previous_questions)}
"""

            result = self._generate_structured(
                system_instruction=QUESTION_SYSTEM_PROMPT,
                prompt=prompt,
                response_schema=QuestionResult,
            )

            if result is None:
                continue

            candidate = _clean_text(result.question)
            if not candidate:
                continue

            if self._is_duplicate_question(candidate, previous_questions):
                logger.warning(
                    "Gemini generated duplicate question on attempt %s: %s",
                    attempt + 1,
                    candidate,
                )
                continue

            return result

        logger.warning("Gemini could not generate a unique question after retries; using curated bank.")
        bank_result = self._question_bank_fallback(
            domain=domain,
            difficulty=difficulty,
            interview_type=interview_type,
            previous_questions=previous_questions,
            turns=turns,
        )
        if bank_result is not None:
            return bank_result

        # The bank is intentionally larger than the maximum interview length.
        # This final guard should therefore be unreachable in normal use.
        return QuestionResult(
            question=(
                f"Explain one important concept in {domain} that a {role} candidate should know, "
                "and give a practical example."
            ),
            source="safe-fallback",
            fallback_reason="All curated questions were exhausted; safe generic fallback used.",
        )

    # ----------------------------------------------------------------
    # Final AI interview assessment
    # ----------------------------------------------------------------

    @staticmethod
    def _local_generate_report(
        role: str,
        experience: str,
        interview_type: str,
        difficulty: str,
        reviews: List[Dict[str, Any]],
        overall_score: int,
    ) -> ReportResult:
        """Generate a useful final report without requiring Gemini.

        This is intentionally evidence-driven. It uses the scores and text
        already produced during the interview and never invents candidate
        experience or skills. The report is therefore still useful when the
        Gemini quota is exhausted.
        """
        if not reviews:
            return ReportResult(
                summary="No interview answers were available for assessment.",
                strongest_skills=[],
                skill_gaps=["Complete the interview to generate evidence-based feedback."],
                technical_assessment="No technical evidence was available.",
                communication_assessment="No communication evidence was available.",
                problem_solving_assessment="No problem-solving evidence was available.",
                role_readiness="Not enough evidence",
                improvement_plan=["Complete a full interview and submit answers for assessment."],
                seven_day_plan=[
                    "Day 1: Review the role requirements.",
                    "Day 2: Practice core technical concepts.",
                    "Day 3: Solve two practical problems.",
                    "Day 4: Practice explaining project decisions.",
                    "Day 5: Practice role-specific questions.",
                    "Day 6: Take a timed mock interview.",
                    "Day 7: Review mistakes and repeat weak topics.",
                ],
                recommendation="Complete another interview after targeted preparation.",
            )

        def score_values(key: str) -> List[float]:
            values = []
            for review in reviews:
                try:
                    value = float((review.get("evaluation") or {}).get(key))
                    values.append(max(0.0, min(10.0, value)))
                except (TypeError, ValueError):
                    pass
            return values

        def avg(key: str) -> float:
            values = score_values(key)
            return sum(values) / len(values) if values else 0.0

        dimension_map = {
            "correctness": "technical correctness",
            "technical_knowledge": "technical knowledge",
            "relevance": "answer relevance",
            "clarity": "clarity",
            "communication": "communication",
        }

        dimension_avgs = {key: avg(key) for key in dimension_map}
        strong_dimensions = [
            label for key, label in dimension_map.items()
            if dimension_avgs[key] >= 7.5
        ]
        weak_dimensions = [
            label for key, label in dimension_map.items()
            if dimension_avgs[key] <= 5.5
        ]

        if not strong_dimensions:
            best_key = max(dimension_avgs, key=dimension_avgs.get)
            strong_dimensions = [dimension_map[best_key]]
        if not weak_dimensions:
            worst_key = min(dimension_avgs, key=dimension_avgs.get)
            weak_dimensions = [dimension_map[worst_key]]

        # Pull only concepts actually mentioned in interview questions.
        topic_catalog = [
            "Python", "FastAPI", "Django", "REST APIs", "MySQL", "SQLAlchemy",
            "Docker", "AWS", "Node.js", "Express.js", "SQL", "Git", "GitHub",
            "Pandas", "NumPy", "machine learning", "data analysis", "authentication",
            "API design", "error handling", "database design", "testing",
        ]
        observed_topics = []
        all_question_text = " ".join(
            str(review.get("question", "")) for review in reviews
        ).lower()
        for topic in topic_catalog:
            if topic.lower() in all_question_text:
                observed_topics.append(topic)

        # Use per-answer strengths/weaknesses/missing points as evidence.
        strengths = []
        gaps = []
        tips = []
        for review in reviews:
            evaluation = review.get("evaluation") or {}
            for item in evaluation.get("strengths", []) or []:
                text = _clean_text(item)
                if text and text not in strengths:
                    strengths.append(text)
            for item in (
                list(evaluation.get("weaknesses", []) or [])
                + list(evaluation.get("missing_points", []) or [])
            ):
                text = _clean_text(item)
                if text and text not in gaps:
                    gaps.append(text)
            for item in evaluation.get("improvement_tips", []) or []:
                text = _clean_text(item)
                if text and text not in tips:
                    tips.append(text)

        # Keep the report readable rather than dumping every evaluator note.
        strengths = strengths[:6]
        gaps = gaps[:6]
        tips = tips[:6]

        if overall_score >= 80:
            readiness = "Strong for the selected level; continue targeted practice on remaining gaps."
        elif overall_score >= 65:
            readiness = "Developing well; targeted practice is recommended before the next interview."
        elif overall_score >= 50:
            readiness = "Developing; strengthen the weaker technical areas before the next interview."
        else:
            readiness = "Needs significant targeted preparation before the next interview."

        observed_text = ", ".join(observed_topics[:6]) if observed_topics else "the topics covered"
        strong_text = ", ".join(strong_dimensions)
        weak_text = ", ".join(weak_dimensions)

        summary = (
            f"The candidate completed a {difficulty.lower()} {interview_type.lower()} interview "
            f"for the {role} role and achieved {overall_score}/100. "
            f"The strongest evidence was in {strong_text}, while the main improvement area "
            f"was {weak_text}. The interview covered {observed_text}."
        )

        technical_assessment = (
            f"Technical performance averaged {avg('technical_knowledge'):.1f}/10 for technical "
            f"knowledge and {avg('correctness'):.1f}/10 for correctness. "
            + (
                "The answers showed useful understanding of the concepts discussed."
                if avg("technical_knowledge") >= 7
                else "Several technical concepts need clearer explanations and more practical examples."
            )
        )

        communication_assessment = (
            f"Clarity averaged {avg('clarity'):.1f}/10 and communication averaged "
            f"{avg('communication'):.1f}/10. "
            + (
                "Answers were generally clear enough to communicate the technical approach."
                if avg("communication") >= 7
                else "Use a more structured answer format: approach, implementation, trade-offs, and result."
            )
        )

        problem_solving_assessment = (
            f"Relevance averaged {avg('relevance'):.1f}/10. "
            + (
                "The candidate generally connected answers to the problem being asked."
                if avg("relevance") >= 7
                else "Focus first on the exact problem, then explain the solution step-by-step with a concrete example."
            )
        )

        improvement_plan = tips or [
            "Review the concepts that received the lowest evaluation scores.",
            "Practice explaining technical decisions with concrete examples.",
            "Solve practical role-specific problems under a time limit.",
        ]

        while len(improvement_plan) < 3:
            improvement_plan.append("Repeat a targeted mock interview and compare the new scores.")

        seven_day_plan = [
            "Day 1: Review the interview questions and identify the three weakest concepts.",
            "Day 2: Study those concepts and write short explanations in your own words.",
            "Day 3: Build or solve one practical task related to the weakest technical topic.",
            "Day 4: Practice role-specific questions and explain your project decisions aloud.",
            "Day 5: Complete a timed Coding/SQL or technical practice session.",
            "Day 6: Take another mock interview and focus on the previous weak areas.",
            "Day 7: Review the new scores, remaining gaps, and create the next practice target.",
        ]

        recommendation = (
            "Continue with another targeted mock interview after practicing the identified gaps. "
            "Prioritize concepts that were both important to the role and weak in this session."
        )

        return ReportResult(
            summary=summary,
            strongest_skills=(strengths[:5] if strengths else strong_dimensions[:5]),
            skill_gaps=(gaps[:5] if gaps else weak_dimensions[:5]),
            technical_assessment=technical_assessment,
            communication_assessment=communication_assessment,
            problem_solving_assessment=problem_solving_assessment,
            role_readiness=readiness,
            improvement_plan=improvement_plan[:6],
            seven_day_plan=seven_day_plan,
            recommendation=recommendation,
        )

    def generate_report(
        self,
        role: str,
        experience: str,
        interview_type: str,
        difficulty: str,
        reviews: List[Dict[str, Any]],
        overall_score: int,
    ) -> Optional[ReportResult]:
        """Generate the final report with Gemini when available and a
        deterministic evidence-based report when Gemini is unavailable."""
        if not reviews:
            return self._local_generate_report(
                role, experience, interview_type, difficulty, reviews, overall_score
            )

        interview_data = []
        for index, review in enumerate(reviews, start=1):
            evaluation = review.get("evaluation", {}) or {}
            interview_data.append(
                {
                    "question_number": index,
                    "question": review.get("question", ""),
                    "candidate_answer": review.get("answer", ""),
                    "evaluation": {
                        key: _safe_int(evaluation.get(key))
                        for key in (
                            "correctness", "technical_knowledge", "relevance",
                            "clarity", "communication", "overall_score",
                        )
                    }
                    | {
                        "strengths": evaluation.get("strengths", []),
                        "weaknesses": evaluation.get("weaknesses", []),
                        "missing_points": evaluation.get("missing_points", []),
                        "improvement_tips": evaluation.get("improvement_tips", []),
                    },
                }
            )

        if self.configured:
            prompt = f"""
Candidate interview profile:

Role: {role}
Experience level: {experience}
Interview type: {interview_type}
Difficulty: {difficulty}
Overall interview score: {overall_score}/100

The numerical evaluations below have already been generated from
individual answers. Treat those evaluations as evidence.

INTERVIEW DATA:

{json.dumps(interview_data, indent=2, ensure_ascii=False)}

Now produce the final candidate assessment.

Important:
- Strong skills must be skills actually demonstrated.
- Skill gaps must come from repeated weaknesses, missing concepts,
  low scores, or incorrect answers.
- Technical assessment must reference actual technical performance.
- Communication assessment must use the clarity/communication evidence.
- Problem-solving assessment should only make claims supported by the interview questions and answers.
- Role readiness must be realistic for the selected experience level.
- The 7-day plan must target the actual weaknesses found in this interview.
- Do not claim that the candidate knows something merely because the interview question mentioned it.
- Do not invent work experience or projects.
- Avoid generic motivational statements.
- Make every recommendation actionable.

Return the exact structured response requested by the schema.
""".strip()

            result = self._generate_structured(
                system_instruction=REPORT_SYSTEM_PROMPT,
                prompt=prompt,
                response_schema=ReportResult,
            )
            if result is not None:
                return result

        logger.info("Using deterministic final report because Gemini is unavailable or quota-limited.")
        return self._local_generate_report(
            role, experience, interview_type, difficulty, reviews, overall_score
        )

    # ----------------------------------------------------------------
    # Fallback question
    # ----------------------------------------------------------------

    @classmethod
    def _personalized_fallback_question(
        cls,
        role: str,
        index: int,
        resume_analysis: Optional[Dict[str, Any]] = None,
        jd_analysis: Optional[Dict[str, Any]] = None,
        jd_match: Optional[Dict[str, Any]] = None,
        interview_type: str = "technical",
        previous_questions: Optional[List[str]] = None,
        turns: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Deterministic adaptive interviewer used when Gemini is unavailable.

        The fallback is deliberately evidence-driven: it uses JD gaps first,
        then matched skills/projects, and avoids questions already asked.
        This keeps the interview personalized even when the Gemini quota is
        exhausted.
        """
        resume_analysis = resume_analysis or {}
        jd_analysis = jd_analysis or {}
        jd_match = jd_match or {}
        previous_questions = previous_questions or []
        turns = turns or []

        def as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [_clean_text(x) for x in value if _clean_text(x)]
            cleaned = _clean_text(value)
            return [cleaned] if cleaned else []

        def unique(items: List[str]) -> List[str]:
            seen = set()
            result = []
            for item in items:
                key = re.sub(r"\s+", " ", item.lower()).strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        previous = [q for q in previous_questions if _clean_text(q)]
        projects = as_list(resume_analysis.get("projects"))
        resume_skills = unique(
            as_list(resume_analysis.get("technical_skills"))
            + as_list(resume_analysis.get("data_analytics_skills"))
            + as_list(resume_analysis.get("web_development_skills"))
            + as_list(resume_analysis.get("tools_and_technologies"))
        )
        missing = unique(as_list(jd_match.get("missing_skills")))
        partial = unique(as_list(jd_match.get("partially_matched_skills")))
        matched = unique(as_list(jd_match.get("matched_skills")))
        topics = unique(as_list(jd_analysis.get("important_interview_topics")))
        responsibilities = as_list(jd_analysis.get("responsibilities"))
        languages = unique(as_list(jd_analysis.get("programming_languages")))
        frameworks = unique(as_list(jd_analysis.get("frameworks_and_libraries")))
        databases = unique(as_list(jd_analysis.get("databases")))

        itype = (interview_type or "technical").lower().strip()
        role_text = role or "this role"
        candidates: List[str] = []

        if "coding" in itype or "programming" in itype:
            for skill in missing + partial + languages + frameworks:
                candidates.append(
                    f"Coding question: Write a practical solution using {skill} for a problem relevant to a {role_text} role. Explain your approach, time complexity, and edge cases."
                )
            candidates += [
                "Write a Python function to find the first non-repeating character in a string. Explain the time and space complexity.",
                "Write a Python function to remove duplicate values from a list while preserving order. Explain your approach.",
                "Given a list of integers, find the two values whose sum equals a target. Explain the data structure you would use and its complexity.",
            ]

        elif "sql" in itype or "database" in itype:
            db = databases[0] if databases else "MySQL"
            candidates += [
                f"Using {db}, write a query to find the top three customers by total purchase amount. Explain how you would handle ties.",
                f"In {db}, explain the difference between INNER JOIN and LEFT JOIN and give a practical example.",
                f"How would you optimize a slow query in {db}? Discuss indexes, EXPLAIN, and query design.",
                "Write a SQL query using GROUP BY and HAVING to identify groups with more than five records.",
            ]

        # Coding/SQL mode is handled before generic JD questions.
        # This section remains as a secondary fallback only.

        # Answer-driven adaptation. Weak answers get a remediation question
        # first; strong answers advance to the next uncovered requirement.
        if last_turn and last_score is not None and last_score < 5:
            last_question = _clean_text(last_turn.get("question"))
            if last_question:
                candidates.append(
                    f"Your previous answer showed a gap on this topic: {last_question}\n"
                    "Re-test the same concept with a simpler practical scenario. Ask the candidate "
                    "to explain the correct approach step by step and mention one common mistake to avoid."
                )

        # Adaptive order: missing -> partial -> project -> matched -> JD topic.
        # When the previous answer was strong, this order intentionally moves
        # away from the previous topic and into the next JD gap.
        for skill in missing:
            candidates.append(
                f"The job description requires {skill}. Explain what {skill} is, how it works, and how you would use it in a practical project for this role. Include one implementation detail or trade-off."
            )
        for skill in partial:
            candidates.append(
                f"You have related experience but not direct evidence of {skill}. Explain how your existing experience could transfer to {skill}, and describe one practical implementation you would build."
            )
        for project in projects:
            name = project.split(":", 1)[0].strip()
            if name:
                candidates.append(
                    f"You mentioned {name} on your resume. What was the most difficult technical problem in that project, and how did you solve it?"
                )
        for skill in matched:
            candidates.append(
                f"You list {skill} and it is relevant to this role. Describe a real project or task where you used {skill}, including one challenge and how you handled it."
            )
        for responsibility in responsibilities:
            candidates.append(
                f"This role involves {responsibility.lower()}. Walk me through how you would perform that task from requirements to testing."
            )
        for topic in topics:
            candidates.append(
                f"For a {role_text} position, explain the key concepts behind {topic} and give a practical example."
            )

        # Keep one technology-focused question even if structured fields are sparse.
        for skill in frameworks + languages + databases:
            candidates.append(
                f"Explain one important concept in {skill} that a candidate for a {role_text} position should understand, and give a practical example."
            )

        for candidate in candidates:
            if not cls._is_duplicate_question(candidate, previous):
                return candidate

        return cls._fallback_question(role, index)

    @staticmethod
    def _fallback_question(
        role: str,
        index: int,
    ) -> str:

        role_lower = role.lower()

        python_questions = [
            "What is the difference between a list and a tuple in Python?",
            "How does exception handling work in Python?",
            "What is the purpose of a Python dictionary?",
            "Explain the difference between append() and extend().",
            "What are Python decorators and when would you use them?",
            "What is the difference between shallow copy and deep copy?",
            "How does a context manager work in Python?",
            "What is the difference between == and is in Python?",
        ]

        general_questions = [
            f"What are the most important technical skills for a {role}?",
            "Explain one technical problem you would expect in this role and how you would solve it.",
            "How would you debug a program that produces an unexpected result?",
            "How would you improve the performance of a slow application?",
            "Explain a technical concept from your field using a simple example.",
            "What trade-offs would you consider when choosing between two technical approaches?",
            "How would you test a feature before considering it complete?",
            "Describe a structured approach to solving a difficult technical problem.",
        ]

        if "python" in role_lower:

            return python_questions[
                index % len(python_questions)
            ]

        return general_questions[
            index % len(general_questions)
        ]


# -------------------------------------------------------------------
# Answer guardrail
# -------------------------------------------------------------------

def answer_guardrail(
    answer: str,
) -> tuple[bool, str]:

    text = answer.strip()

    if len(text) < 2:

        return (
            False,
            "The answer is too short to evaluate meaningfully.",
        )

    normalized = re.sub(
        r"\s+",
        " ",
        text.lower(),
    )

    invalid_answers = {
        "",
        ".",
        "..",
        "...",
        ";",
        "-",
        "na",
        "n/a",
        "nothing",
        "i don't know",
        "idk",
        "don't know",
        "no idea",
    }

    if normalized in invalid_answers:

        return (
            False,
            "The response does not contain a meaningful answer.",
        )

    return True, ""


# -------------------------------------------------------------------
# Fallback evaluation
# -------------------------------------------------------------------

def fallback_evaluation(
    reason: str,
    source: str = "mock",
) -> EvaluationResult:

    return EvaluationResult(
        correctness=0,
        technical_knowledge=0,
        relevance=0,
        clarity=0,
        communication=0,
        overall_score=0,
        strengths=[],
        weaknesses=[reason],
        missing_points=[],
        improvement_tips=[
            "Answer the exact question directly.",
            "Include the relevant technical concept.",
            "Use a short concrete example when appropriate.",
        ],
        ideal_answer=(
            "Provide a concise and technically correct answer."
        ),
        source=source,
        fallback_reason=reason,
    )