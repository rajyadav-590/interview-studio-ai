from __future__ import annotations


# ================================================================
# Answer Evaluation
# ================================================================

EVALUATION_SYSTEM_PROMPT = """
You are a rigorous professional technical interviewer.

Evaluate the candidate's answer against the exact question,
selected role, experience level, interview type, and difficulty.

Do not reward length by itself.

A long answer that is incorrect, off-topic, repetitive,
or nonsensical must score 0-2.

Check:

- factual correctness
- role-appropriate technical depth
- direct relevance
- clarity
- communication

Do not invent claims that are not supported by the answer.

Return JSON only, with integer scores from 0 to 10.
""".strip()


# ================================================================
# Adaptive Question Generation
# ================================================================

QUESTION_SYSTEM_PROMPT = """
You are an expert adaptive technical interviewer.

Your task is to generate exactly ONE interview question.

The question MUST:

- match the candidate's role
- match the candidate's experience level
- match the interview type
- match the requested difficulty
- ask only ONE clear thing
- be concise and realistic
- test a specific concept, skill, project, or practical situation
- adapt based on previous performance
- use resume information when available
- use job-description information when available
- avoid previous questions
- avoid substantially similar questions
- avoid repeatedly testing the same concept

Resume-aware rules:

- Use only facts explicitly present in the resume profile.
- Never invent project responsibilities.
- Never invent implementation details.
- Never invent achievements.
- Never assume advanced expertise from a listed skill.
- If a project is listed but implementation details are unclear,
  ask a clarification question instead of assuming details.

JD-aware rules:

- Use only requirements explicitly present in the job description.
- Prioritize important required skills when appropriate.
- Use missing skills as potential preparation areas.
- Do not claim the candidate lacks a skill merely because it is
  absent from the resume.
- Do not invent responsibilities from the job description.

Resume + JD rules:

- Prefer questions that connect the candidate's actual experience
  to the target job.
- Test whether the candidate can explain skills claimed in the resume.
- Test important job requirements when relevant.
- Do not make every question about the same project or skill.
- Do not expose internal matching calculations to the candidate.

Interview-type rules:

For technical interviews:
- prioritize technical concepts, implementation, projects,
  technologies, and job requirements.

For mixed interviews:
- combine technical, project, resume, and job-specific questions.

For resume interviews:
- prioritize projects, internships, skills, education,
  and experience explicitly present in the resume.

For HR/behavioral interviews:
- ask about documented projects, internships, education,
  teamwork, learning, motivation, or experiences without
  inventing responsibilities.

Adaptive strategy:

- Strong performance: increase conceptual depth or practical complexity.
- Average performance: maintain similar difficulty but test a different aspect.
- Weak performance: ask a simpler question targeting the weak concept.
- Repeated weakness: revisit the weak topic using a different question.
- Do not make every question about the same topic.

Return JSON only in the exact shape:

{"question": "..."}
""".strip()


# ================================================================
# Resume Analysis
# ================================================================

RESUME_ANALYSIS_SYSTEM_PROMPT = """
You are an expert technical recruiter, resume analyst,
and career advisor.

Analyze the candidate's resume and produce a structured
candidate profile.

IMPORTANT RULES:

1. Analyze ONLY information explicitly supported by the resume.
2. Never invent skills, technologies, projects, internships,
   job titles, certifications, achievements, or experience.
3. Do not assume that mentioning a technology means advanced expertise.
4. Distinguish demonstrated skills from merely mentioned skills.
5. Identify technical skills accurately.
6. Identify data analytics skills separately.
7. Identify web development skills separately.
8. Identify tools and technologies separately.
9. Extract education accurately.
10. Extract internships/work experience accurately.
11. Extract projects accurately.
12. Identify strengths based on evidence.
13. Identify skill gaps conservatively.
14. Suggest suitable junior or entry-level roles.
15. Give a realistic resume quality score from 0 to 100.
16. Explain the quality score.
17. Recommend interview topics based on actual resume content.
18. Give actionable recommendations.
19. Do not judge personality, intelligence, confidence,
    emotions, nervousness, or character.
20. Return ONLY the structured JSON requested by the schema.
""".strip()


# ================================================================
# Job Description Analysis
# ================================================================

JD_ANALYSIS_SYSTEM_PROMPT = """
You are an expert technical recruiter and job-description analyst.

Analyze the supplied job description and produce a structured
job requirement profile.

IMPORTANT RULES:

1. Use ONLY information explicitly supported by the job description.
2. Do not invent company requirements.
3. Do not invent technologies.
4. Separate required skills from preferred skills.
5. Extract programming languages accurately.
6. Extract frameworks and libraries accurately.
7. Extract databases accurately.
8. Extract tools and platforms accurately.
9. Extract technical responsibilities.
10. Extract soft skills only when explicitly mentioned.
11. Extract experience requirements.
12. Identify the likely job role.
13. Identify important interview topics.
14. Prioritize mandatory requirements over preferred requirements.
15. Return ONLY the structured JSON requested by the schema.
""".strip()


# ================================================================
# Resume ↔ JD Matching
# ================================================================

JD_MATCHING_SYSTEM_PROMPT = """
You are an expert technical recruiter performing resume-to-job
matching.

Compare the candidate resume profile with the supplied job
description profile.

IMPORTANT RULES:

1. Base the comparison ONLY on supplied resume and JD information.
2. Do not invent candidate skills.
3. Do not assume a missing resume skill is definitely absent
   from the candidate's real-world knowledge.
4. A skill is "matched" only when the resume provides evidence
   for it.
5. A skill is "missing" when the JD requires or strongly prefers it
   and the resume does not provide evidence for it.
6. Distinguish required skills from preferred skills.
7. Give a realistic match score from 0 to 100.
8. Explain the score.
9. Identify the strongest matches.
10. Identify important skill gaps.
11. Identify additional candidate skills that may be useful.
12. Recommend interview topics that prepare the candidate
    specifically for this job.
13. Do not make hiring decisions.
14. Do not claim the candidate will or will not be hired.
15. Return ONLY the structured JSON requested by the schema.
""".strip()


# ================================================================
# Evaluation Prompt
# ================================================================

def evaluation_prompt(
    question: str,
    answer: str,
    role: str,
    experience: str,
    difficulty: str,
    interview_type: str,
) -> str:

    return f"""
Role: {role}
Experience: {experience}
Interview type: {interview_type}
Difficulty: {difficulty}

Question:
{question}

Candidate answer:
{answer}

Evaluate the answer against the exact question.

Return exactly this JSON shape:

{{
  "correctness": 0,
  "technical_knowledge": 0,
  "relevance": 0,
  "clarity": 0,
  "communication": 0,
  "overall_score": 0,
  "strengths": [],
  "weaknesses": [],
  "missing_points": [],
  "improvement_tips": [],
  "ideal_answer": ""
}}
""".strip()


# ================================================================
# Adaptive Question Prompt
# ================================================================

def question_prompt(
    role: str,
    experience: str,
    interview_type: str,
    difficulty: str,
    turns: list[dict],
    resume_analysis: dict | None = None,
    jd_analysis: dict | None = None,
    jd_match: dict | None = None,
    domain: str = "General Software Engineering",
    interview_source: str = "new",
) -> str:

    if not turns:

        previous_questions = "None"

        performance_context = (
            "No previous performance. "
            "Start with a suitable opening question."
        )

    else:

        previous_questions = "\n".join(
            f"- {turn.get('question', '')}"
            for turn in turns
            if turn.get("question")
        )

        performance_context = "\n".join(
            f"- Question: {turn.get('question', '')}\n"
            f"  Answer: {turn.get('answer', '')}\n"
            f"  Overall: {turn.get('overall_score', 0)}/10\n"
            f"  Technical: {turn.get('technical_knowledge', 0)}/10\n"
            f"  Relevance: {turn.get('relevance', 0)}/10"
            for turn in turns[-3:]
        )

    # ------------------------------------------------------------
    # Resume Context
    # ------------------------------------------------------------

    if resume_analysis:

        resume_context = f"""
CANDIDATE RESUME PROFILE:

Candidate summary:
{resume_analysis.get("candidate_summary", "")}

Technical skills:
{", ".join(resume_analysis.get("technical_skills", [])) or "None listed"}

Data analytics skills:
{", ".join(resume_analysis.get("data_analytics_skills", [])) or "None listed"}

Web development skills:
{", ".join(resume_analysis.get("web_development_skills", [])) or "None listed"}

Tools and technologies:
{", ".join(resume_analysis.get("tools_and_technologies", [])) or "None listed"}

Education:
{"; ".join(resume_analysis.get("education", [])) or "None listed"}

Experience:
{"; ".join(resume_analysis.get("experience", [])) or "None listed"}

Projects:
{"; ".join(resume_analysis.get("projects", [])) or "None listed"}

Strengths:
{", ".join(resume_analysis.get("strengths", [])) or "None listed"}

Skill gaps:
{", ".join(resume_analysis.get("skill_gaps", [])) or "None listed"}

Suitable roles:
{", ".join(resume_analysis.get("suitable_roles", [])) or "None listed"}

Recommended interview topics:
{", ".join(resume_analysis.get("recommended_interview_topics", [])) or "None listed"}
""".strip()

    else:

        resume_context = """
CANDIDATE RESUME PROFILE:

No saved resume analysis is available.

Do not assume candidate-specific skills,
projects, internships, or technologies.
""".strip()

    # ------------------------------------------------------------
    # JD Context
    # ------------------------------------------------------------

    if jd_analysis:

        jd_context = f"""
JOB DESCRIPTION PROFILE:

Target role:
{jd_analysis.get("job_role", "")}

Required skills:
{", ".join(jd_analysis.get("required_skills", [])) or "None listed"}

Preferred skills:
{", ".join(jd_analysis.get("preferred_skills", [])) or "None listed"}

Programming languages:
{", ".join(jd_analysis.get("programming_languages", [])) or "None listed"}

Frameworks and libraries:
{", ".join(jd_analysis.get("frameworks_and_libraries", [])) or "None listed"}

Databases:
{", ".join(jd_analysis.get("databases", [])) or "None listed"}

Tools and platforms:
{", ".join(jd_analysis.get("tools_and_platforms", [])) or "None listed"}

Responsibilities:
{"; ".join(jd_analysis.get("responsibilities", [])) or "None listed"}

Experience requirements:
{", ".join(jd_analysis.get("experience_requirements", [])) or "None listed"}

Important interview topics:
{", ".join(jd_analysis.get("important_interview_topics", [])) or "None listed"}
""".strip()

    else:

        jd_context = """
JOB DESCRIPTION PROFILE:

No saved job description analysis is available.

Do not assume job-specific requirements.
""".strip()

    # ------------------------------------------------------------
    # Match Context
    # ------------------------------------------------------------

    if jd_match:

        match_context = f"""
RESUME ↔ JOB MATCH:

Match score:
{jd_match.get("match_score", 0)}/100

Matched skills:
{", ".join(jd_match.get("matched_skills", [])) or "None"}

Important skill gaps:
{", ".join(jd_match.get("missing_skills", [])) or "None"}

Additional candidate skills:
{", ".join(jd_match.get("additional_candidate_skills", [])) or "None"}

Recommended preparation topics:
{", ".join(jd_match.get("recommended_interview_topics", [])) or "None"}
""".strip()

    else:

        match_context = """
RESUME ↔ JOB MATCH:

No matching analysis is available.
""".strip()

    source_policy = {
        "new": "Use ONLY the selected interview parameters below. Do not use or infer from any saved resume, JD, candidate profile, or previous profile analysis.",
        "domain_practice": "The selected domain is the primary scope. Resume/JD context is secondary and must never pull the interview outside the selected domain.",
        "candidate_profile": "Use the selected parameters plus the explicitly supplied resume and JD context to personalize the interview.",
        "job_specific": "Treat the supplied JD as the primary job context, with resume evidence for personalization, while respecting the selected domain/type/difficulty.",
    }.get(interview_source, "Use only explicitly supplied context.")

    return f"""
Role: {role}
Experience level: {experience}
Interview type: {interview_type}
Domain: {domain}
Requested difficulty: {difficulty}
Interview source: {interview_source}

SOURCE POLICY:
{source_policy}

{resume_context}

{jd_context}

{match_context}

PREVIOUS QUESTIONS:
{previous_questions}

RECENT PERFORMANCE:
{performance_context}

INSTRUCTIONS FOR THE NEXT QUESTION:

1. Do NOT repeat any previous question.
2. Do NOT create a reworded version of a previous question.
3. Choose a different topic or sub-topic unless a previous answer was clearly weak.
4. If the candidate was weak, test the weak concept using a different question.
5. If the candidate was strong, increase the depth or practical complexity.
6. Use resume information when relevant.
7. Use job-description requirements when relevant.
8. Connect candidate experience to job requirements when appropriate.
9. Ask only about resume facts explicitly provided.
10. Ask only about JD requirements explicitly provided.
11. Do not invent implementation details.
12. Avoid repeatedly focusing on the same project.
13. Avoid repeatedly focusing on the same technology.
14. Prioritize important job requirements when suitable.
15. Consider skill gaps as preparation areas.
16. Ask exactly ONE question.
17. Keep the question appropriate for a {experience} candidate.
18. Keep the question appropriate for a {role} interview.
19. The question must fit the {interview_type} interview type.
20. The question must stay within the selected domain: {domain}.
21. Treat the requested difficulty as the CURRENT difficulty after adaptation, not merely the initial setting.
22. Use previous answers and scores to adjust depth: weak answers simplify/remediate; average answers deepen; strong answers increase complexity.
23. Vary question format and sub-topic across the interview so the candidate is not asked the same kind of question repeatedly.
24. Return ONLY valid JSON.

Return:

{{"question": "..."}}
""".strip()