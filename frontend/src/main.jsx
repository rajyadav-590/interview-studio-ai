import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  LogOut,
  Menu,
  Play,
  Sparkles,
  Target,
  X,
  TrendingUp,
  AlertCircle,
  Lightbulb,
  CalendarDays,
  Award,
  MessageSquare,
  Code2,
  ChevronDown,
  ChevronUp,
  UploadCloud,
  FileText,
  BriefcaseBusiness,
  RefreshCw,
  Activity,
  Mic2,
  Brain,
} from 'lucide-react';
import './styles.css';

const setupEnhancementStyles = `
  .profile-context-card {
    display: flex;
    justify-content: space-between;
    gap: 28px;
    align-items: center;
    margin: 0 auto 24px;
    max-width: 980px;
    padding: 22px 24px;
    border: 1px solid rgba(35, 48, 64, 0.12);
    background: rgba(255,255,255,0.78);
    border-radius: 18px;
  }

  .profile-context-card h3 {
    margin: 7px 0 6px;
  }

  .profile-context-card p {
    margin: 0;
    max-width: 650px;
    line-height: 1.55;
  }

  .profile-context-stats {
    display: flex;
    gap: 18px;
    flex-shrink: 0;
  }

  .profile-context-stats div {
    min-width: 72px;
    text-align: center;
  }

  .profile-context-stats span {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
    opacity: .62;
  }

  .profile-context-stats strong {
    display: block;
    margin-top: 5px;
    font-size: 20px;
  }

  .field-hint {
    display: block;
    margin-top: 7px;
    font-size: 12px;
    line-height: 1.4;
    opacity: .68;
  }

  .setup-topics {
    margin-top: 4px;
    padding-top: 4px;
  }

  .interview-mode-badge {
    display: flex;
    gap: 10px;
    align-items: center;
    margin: 12px 0 16px;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(35, 48, 64, 0.05);
    font-size: 12px;
  }

  .interview-mode-badge strong {
    letter-spacing: .08em;
    font-size: 10px;
  }

  .interview-mode-badge span {
    opacity: .68;
  }

  .answer-editor.code-editor {
    min-height: 260px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 14px;
    line-height: 1.55;
    tab-size: 4;
    white-space: pre;
  }

  @media (max-width: 760px) {
    .profile-context-card {
      flex-direction: column;
      align-items: flex-start;
    }

    .profile-context-stats {
      width: 100%;
      justify-content: space-between;
    }
  }
`;

if (typeof document !== 'undefined' && !document.getElementById('setup-enhancement-styles')) {
  const style = document.createElement('style');
  style.id = 'setup-enhancement-styles';
  style.textContent = setupEnhancementStyles;
  document.head.appendChild(style);
}


const API = import.meta.env.VITE_API_URL || '/api';

const demoInterviews = [
  {
    id: 'demo',
    role: 'Frontend Developer',
    interview_type: 'Technical',
    score: 84,
    created_at: '2026-09-04',
    status: 'completed',
    total_questions: 10,
    answered: 10,
  },
];

const initialSetup = {
  role: 'Software Engineer',
  experience: '1–2 Years',
  interview_type: 'Mixed',
  difficulty: 'Medium',
  interview_source: 'new',
  domain: 'General Software Engineering',
  total_questions: 5,
  video_mode: true,
};

async function request(path, options = {}) {
  const { token, ...fetchOptions } = options;

  const isFormData = fetchOptions.body instanceof FormData;

  const response = await fetch(`${API}${path}`, {
    headers: {
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...(token
        ? {
            Authorization: `Bearer ${token}`,
          }
        : {}),
    },
    ...fetchOptions,
  });

  const body = await response.text();

  let payload = null;

  try {
    payload = body ? JSON.parse(body) : null;
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(
      payload?.detail || body || `Request failed (${response.status})`
    );
  }

  return payload;
}

function inferInterviewDomain(role, jd = {}, resume = {}) {
  const text = [
    role,
    ...(Array.isArray(jd.important_interview_topics) ? jd.important_interview_topics : []),
    ...(Array.isArray(jd.required_skills) ? jd.required_skills : []),
    ...(Array.isArray(jd.frameworks_and_libraries) ? jd.frameworks_and_libraries : []),
  ].join(' ').toLowerCase();

  const rules = [
    ['Machine Learning', ['machine learning', 'scikit-learn', 'ml model']],
    ['Deep Learning', ['deep learning', 'tensorflow', 'pytorch', 'neural network']],
    ['Data Science', ['data scientist', 'data science', 'statistics', 'pandas', 'numpy']],
    ['Data Analytics', ['data analyst', 'data analytics', 'power bi', 'excel', 'dashboard']],
    ['AI / NLP', ['nlp', 'natural language', 'llm', 'generative ai']],
    ['Computer Vision', ['computer vision', 'opencv', 'image processing']],
    ['Cybersecurity', ['cybersecurity', 'security analyst', 'penetration testing']],
    ['DevOps / Cloud', ['devops', 'aws', 'azure', 'gcp', 'docker', 'kubernetes']],
    ['Database / SQL', ['sql', 'database', 'mysql', 'postgresql']],
    ['Web Development', ['frontend', 'backend developer', 'full stack', 'react', 'node.js', 'express']],
    ['Python Development', ['python developer']],
  ];

  const match = rules.find(([, terms]) => terms.some((term) => text.includes(term)));
  return match?.[0] || 'Software Development';
}

function inferExperienceLevel(requirements = []) {
  const text = Array.isArray(requirements) ? requirements.join(' ').toLowerCase() : String(requirements || '').toLowerCase();
  if (/5\s*\+|5\s*years|senior|lead/.test(text)) return '5+ Years';
  if (/3\s*[-–]\s*5|3\s*years|4\s*years/.test(text)) return '3–5 Years';
  if (/1\s*[-–]\s*2|1\s*year|2\s*years|junior|associate/.test(text)) return '1–2 Years';
  if (/fresher|entry[- ]level|intern|internship|graduate/.test(text)) return 'Fresher';
  return '';
}

function App() {
  const savedSession = JSON.parse(
    localStorage.getItem('interview-studio-session') || 'null'
  );

  const [view, setView] = useState(
    savedSession?.token ? 'dashboard' : 'home'
  );

  const [user, setUser] = useState(savedSession?.user || null);
  const [token, setToken] = useState(savedSession?.token || '');

  const [setup, setSetup] = useState(initialSetup);
  const [interview, setInterview] = useState(null);
  const [answer, setAnswer] = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [report, setReport] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [llmMode, setLlmMode] = useState('checking');
  const [profile, setProfile] = useState(null);
  const [aiAssessment, setAiAssessment] = useState(null);

  useEffect(() => {
    request('/health')
      .then((result) => setLlmMode(result.mode))
      .catch(() => setLlmMode('offline'));
  }, []);

  const signIn = async (event) => {
    event.preventDefault();

    setLoading(true);
    setError('');

    const data = new FormData(event.currentTarget);

    try {
      const result = await request(
        `/auth/${authMode === 'login' ? 'login' : 'register'}`,
        {
          method: 'POST',
          body: JSON.stringify({
            email: data.get('email'),
            password: data.get('password'),
            name: data.get('name'),
          }),
        }
      );

      setToken(result.token);
      setUser(result.user);

      localStorage.setItem(
        'interview-studio-session',
        JSON.stringify({
          token: result.token,
          user: result.user,
        })
      );

      setView('dashboard');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const startInterview = async (event) => {
    event?.preventDefault();

    setLoading(true);
    setError('');

    try {
      const result = await request('/interviews', {
        method: 'POST',
        body: JSON.stringify(setup),
        token,
      });

      setInterview(result);
      setEvaluation(null);
      setAnswer('');
      setReport(null);
      setView('interview');
    } catch (e) {
      setError(
        token ? e.message : 'Sign in to start a saved interview.'
      );
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (!answer.trim()) {
      return setError('Write a short answer before submitting.');
    }

    setLoading(true);
    setError('');

    try {
      const result = await request(
        `/interviews/${interview.id}/answer`,
        {
          method: 'POST',
          body: JSON.stringify({
            question: interview.next_question,
            answer,
          }),
          token,
        }
      );

      setEvaluation(result.evaluation);
      setLlmMode(result.llm_mode || llmMode);
      setInterview(result.interview);
      setAnswer('');
      if (result.generation_warning) {
        setError(result.generation_warning);
      }

      if (result.interview.status === 'completed') {
        const final = await request(
          `/interviews/${interview.id}/report`,
          {
            token,
          }
        );

        setReport(final);
        setView('report');
      }
    } catch (e) {
      // If the answer was saved but the browser received a transient failure,
      // recover the authoritative server-side interview state instead of
      // allowing the user to resubmit the same question.
      try {
        if (interview?.id) {
          const recovered = await request(`/interviews/${interview.id}`, { token });
          setInterview(recovered);
          setAnswer('');
          setError('The previous submission was recovered from the server. You can continue with the current question.');
          return;
        }
      } catch {
        // Keep the original error when recovery is also unavailable.
      }
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    setToken('');
    setReport(null);
    setInterview(null);

    localStorage.removeItem('interview-studio-session');

    setView('home');
  };

  const nav = (next) => {
    setView(next);
    setMobileOpen(false);
    setError('');
  };

  const startProfileInterview = () => {
    const detectedRole =
      profile?.jd?.analysis?.job_role ||
      profile?.jd?.analysis?.detected_role ||
      profile?.jd?.job_role;

    const inferredDomain = inferInterviewDomain(
      detectedRole || '',
      profile?.jd?.analysis || {},
      profile?.resume?.analysis || {}
    );

    const jdExperience = inferExperienceLevel(
      profile?.jd?.analysis?.experience_requirements || []
    );

    setSetup((current) => ({
      ...current,
      interview_source: 'candidate_profile',
      role: detectedRole || current.role,
      domain: inferredDomain || current.domain,
      experience: jdExperience || current.experience,
    }));

    nav('setup');
  };

  const openReport = async (interviewId) => {
    setLoading(true);
    setError('');

    try {
      const result = await request(
        `/interviews/${interviewId}/report`,
        {
          token,
        }
      );

      setReport(result);
      setView('report');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const openAIInsights = async (interviewId) => {
    setLoading(true);
    setError('');
    try {
      const result = await request(`/interviews/${interviewId}/ai-assessment`, { token });
      setAiAssessment(result);
      setView('ai-insights');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="brand"
          onClick={() => nav('home')}
        >
          <span className="brand-mark">
            <Sparkles size={17} />
          </span>

          interview<span>studio</span>
        </button>

        <button
          className="mobile-menu"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X /> : <Menu />}
        </button>

        <nav className={mobileOpen ? 'nav open' : 'nav'}>
          <button
            onClick={() =>
              nav(user ? 'dashboard' : 'home')
            }
          >
            How it works
          </button>

          {user ? (
            <>
              <button onClick={() => nav('dashboard')}>
                My interviews
              </button>

              <button onClick={() => nav('profile')}>
                Resume & JD
              </button>

              <button
                className="nav-action"
                onClick={() => nav('setup')}
              >
                <Play size={14} />
                New interview
              </button>

              <button
                className="icon-button"
                title="Sign out"
                onClick={logout}
              >
                <LogOut size={16} />
              </button>
            </>
          ) : (
            <button
              className="nav-action"
              onClick={() => nav('auth')}
            >
              Sign in
              <ArrowRight size={14} />
            </button>
          )}
        </nav>
      </header>

      {error && (
        <div className="error-banner">
          {error}
        </div>
      )}

      {llmMode === 'mock-fallback' && (
        <div className="mode-banner">
          Demo AI mode is active. Add `LLM_API_KEY` to `.env`
          for content-based LLM questions and evaluations.
        </div>
      )}

      {view === 'home' && (
        <Home
          onStart={() => nav(user ? 'setup' : 'auth')}
          onAuth={() => nav('auth')}
        />
      )}

      {view === 'auth' && (
        <Auth
          mode={authMode}
          setMode={setAuthMode}
          onSubmit={signIn}
          loading={loading}
          error={error}
        />
      )}

      {view === 'profile' && (
        <ProfileWorkspace
          token={token}
          profile={profile}
          setProfile={setProfile}
          onStartInterview={startProfileInterview}
        />
      )}

      {view === 'dashboard' && (
        <Dashboard
          user={user}
          token={token}
          onNew={() => nav('setup')}
          onReport={openReport}
          onAIInsights={openAIInsights}
        />
      )}

      {view === 'setup' && (
        <Setup
          setup={setup}
          setSetup={setSetup}
          profile={profile}
          onSubmit={startInterview}
          loading={loading}
        />
      )}

      {view === 'interview' && (
        <Interview
          interview={interview}
          answer={answer}
          setAnswer={setAnswer}
          evaluation={evaluation}
          onSubmit={submitAnswer}
          loading={loading}
          llmMode={llmMode}
          token={token}
          videoMode={setup.video_mode}
          onExit={() => nav('dashboard')}
        />
      )}

      {view === 'ai-insights' && (
        <AIInsights assessment={aiAssessment} onBack={() => nav('dashboard')} />
      )}

      {view === 'report' && (
        <Report
          report={report}
          onNew={() => nav('setup')}
        />
      )}
    </div>
  );
}

function Home({ onStart }) {
  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="pulse" />
            Adaptive interview studio · v2
          </div>

          <h1>
            Practice like the
            <br />
            <em>real thing.</em>
          </h1>

          <p className="hero-lead">
            A live interview room that listens for signal,
            adapts to your thinking, and turns every answer
            into your next advantage.
          </p>

          <button
            className="primary-button large"
            onClick={onStart}
          >
            Start a practice interview
            <ArrowRight size={17} />
          </button>

          <div className="proof">
            <div className="avatars">
              <span>AJ</span>
              <span>SK</span>
              <span>MR</span>
            </div>

            <span>
              Built for deliberate practice
              <br />
              <strong>
                from first question to final report
              </strong>
            </span>
          </div>

          <div className="hero-stats">
            <span>
              <strong>6</strong> scoring dimensions
            </span>

            <span>
              <strong>1:1</strong> adaptive follow-ups
            </span>

            <span>
              <strong>24/7</strong> practice room
            </span>
          </div>
        </div>

        <div className="hero-art">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />

          <div className="art-card">
            <div className="art-top">
              <span>LIVE SESSION · ADAPTIVE</span>
              <span className="green-dot" />
            </div>

            <div className="art-question">
              Tell me about a time you changed your mind
              after seeing new data.
            </div>

            <div className="art-meter">
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>

            <div className="art-footer">
              <span>Question 04 / 10</span>
              <span>04:32</span>
            </div>
          </div>

          <div className="float-note">
            <CheckCircle2 size={15} />
            Answer clarity
            <strong>+18%</strong>
          </div>

          <div className="signal-card">
            <span>SESSION SIGNAL</span>

            <strong>
              78<span>/100</span>
            </strong>

            <small>
              +12 from last session
            </small>
          </div>
        </div>
      </section>

      <section className="marquee">
        <span>ADAPTIVE FEEDBACK</span>
        <span>ROLE-SPECIFIC QUESTIONS</span>
        <span>REAL PROGRESS</span>
        <span>CONTENT-BASED SCORING</span>
      </section>

      <section className="feature-section">
        <div className="section-intro">
          <div className="eyebrow">
            The studio approach
          </div>

          <h2>
            Everything you need to
            <br />
            <em>show up sharper.</em>
          </h2>
        </div>

        <div className="feature-grid">
          <Feature
            icon={<BrainCircuit />}
            title="Questions that follow your thinking"
            text="The next question responds to your last answer, so every session feels like a real conversation."
          />

          <Feature
            icon={<Target />}
            title="Feedback you can use"
            text="Get a clear score across clarity, relevance, technical depth, and communication."
          />

          <Feature
            icon={<BarChart3 />}
            title="Progress you can see"
            text="Track patterns across interviews and know exactly where to focus next."
          />
        </div>
      </section>
    </main>
  );
}

function Feature({ icon, title, text }) {
  return (
    <article className="feature">
      <div className="feature-icon">
        {icon}
      </div>

      <h3>{title}</h3>

      <p>{text}</p>

      <ArrowRight size={17} />
    </article>
  );
}

function Auth({
  mode,
  setMode,
  onSubmit,
  loading,
}) {
  return (
    <main className="center-page">
      <div className="auth-card">
        <div className="eyebrow">
          Your private practice room
        </div>

        <h1>
          {mode === 'login'
            ? 'Welcome back.'
            : 'Make space to grow.'}
        </h1>

        <p>
          {mode === 'login'
            ? 'Pick up where your preparation left off.'
            : 'Create your account and start building interview signal.'}
        </p>

        <form onSubmit={onSubmit}>
          {mode === 'register' && (
            <label>
              Name
              <input
                name="name"
                required
                placeholder="Alex Johnson"
              />
            </label>
          )}

          <label>
            Email
            <input
              name="email"
              type="email"
              required
              placeholder="you@example.com"
            />
          </label>

          <label>
            Password
            <input
              name="password"
              type="password"
              minLength="6"
              required
              placeholder="Six characters minimum"
            />
          </label>

          <button
            className="primary-button"
            disabled={loading}
          >
            {loading
              ? 'Opening your studio…'
              : mode === 'login'
              ? 'Sign in'
              : 'Create account'}

            <ArrowRight size={16} />
          </button>
        </form>

        <button
          className="text-button"
          onClick={() =>
            setMode(
              mode === 'login'
                ? 'register'
                : 'login'
            )
          }
        >
          {mode === 'login'
            ? 'Need an account? Create one'
            : 'Already have an account? Sign in'}
        </button>

        <span className="demo-note">
          Without `LLM_API_KEY`, the app stays in explicit
          low-score fallback mode instead of pretending
          evaluations are AI-generated.
        </span>
      </div>
    </main>
  );
}

function Setup({
  setup,
  setSetup,
  profile,
  onSubmit,
  loading,
}) {
  const update = (key, value) => {
    setSetup({
      ...setup,
      [key]: value,
    });
  };

  const detectedRole =
    profile?.jd?.analysis?.job_role ||
    profile?.jd?.analysis?.detected_role ||
    profile?.jd?.job_role ||
    '';

  const matchScore =
    profile?.jd?.resume_match?.match_score ??
    profile?.jd?.match_score ??
    null;

  const requiredSkills = Array.isArray(
    profile?.jd?.analysis?.required_skills
  )
    ? profile.jd.analysis.required_skills
    : [];

  const missingSkills = Array.isArray(
    profile?.jd?.resume_match?.missing_skills
  )
    ? profile.jd.resume_match.missing_skills
    : [];

  const interviewTopics = Array.isArray(
    profile?.jd?.analysis?.important_interview_topics
  )
    ? profile.jd.analysis.important_interview_topics
    : [];

  const roleOptions = [
    'Software Engineer',
    'Frontend Developer',
    'Backend Developer',
    'Full Stack Developer',
    'Python Developer',
    'Data Scientist',
    'AI/ML Engineer',
    'DevOps Engineer',
    ...(detectedRole &&
    ![
      'Software Engineer',
      'Frontend Developer',
      'Backend Developer',
      'Full Stack Developer',
      'Python Developer',
      'Data Scientist',
      'AI/ML Engineer',
      'DevOps Engineer',
    ].includes(detectedRole)
      ? [detectedRole]
      : []),
  ];

  const hasProfile =
    Boolean(profile?.resume?.analysis) ||
    Boolean(profile?.jd?.analysis);

  return (
    <main className="setup-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            New session
          </div>

          <h1>Set the room.</h1>

          <p>
            Shape a session around the role you want next.
          </p>
        </div>

        <div className="step-indicator">
          <span className="active">01</span>
          <i />
          <span>02</span>
          <i />
          <span>03</span>
        </div>
      </div>

      {hasProfile && (
        <section className="profile-context-card">
          <div>
            <div className="eyebrow">
              Personalized interview
            </div>

            <h3>
              {detectedRole
                ? `Preparing for ${detectedRole}`
                : 'Your candidate profile is connected'}
            </h3>

            <p>
              The interviewer will use the connected resume, target JD,
              matched skills, missing skills and previous answers.
              The selected domain and interview settings remain authoritative.
            </p>
          </div>

          <div className="profile-context-stats">
            {matchScore != null && (
              <div>
                <span>JD match</span>
                <strong>{matchScore}/100</strong>
              </div>
            )}

            {requiredSkills.length > 0 && (
              <div>
                <span>JD skills</span>
                <strong>{requiredSkills.length}</strong>
              </div>
            )}

            {missingSkills.length > 0 && (
              <div>
                <span>Skill gaps</span>
                <strong>{missingSkills.length}</strong>
              </div>
            )}
          </div>
        </section>
      )}

      <form
        className="setup-form"
        onSubmit={onSubmit}
      >
        <label>
          Interview source

          <select
            value={setup.interview_source || 'new'}
            onChange={(e) => update('interview_source', e.target.value)}
          >
            <option value="new">New interview — selected parameters only</option>
            <option value="domain_practice">Domain practice — domain first</option>
            <option value="candidate_profile">Candidate profile — resume + JD</option>
            <option value="job_specific">Job-specific — JD first</option>
          </select>

          <small className="field-hint">
            Controls which saved profile context the AI is allowed to use.
          </small>
        </label>

        <label>
          Interview domain

          <select
            value={setup.domain || 'General Software Engineering'}
            onChange={(e) => update('domain', e.target.value)}
          >
            {[
              'Software Development',
              'Web Development',
              'Python Development',
              'Data Structures & Algorithms',
              'Data Science',
              'Machine Learning',
              'Deep Learning',
              'Data Analytics',
              'Database / SQL',
              'AI / NLP',
              'Computer Vision',
              'Cybersecurity',
              'DevOps / Cloud',
              'General Software Engineering',
            ].map((x) => <option key={x}>{x}</option>)}
          </select>

          <small className="field-hint">
            Questions must stay within this domain unless the selected source is explicitly job-specific.
          </small>
        </label>

        <label>
          Role

          <select
            value={setup.role}
            onChange={(e) =>
              update('role', e.target.value)
            }
          >
            {roleOptions.map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>

          {detectedRole && (
            <small className="field-hint">
              ✓ Auto-filled from your analyzed job description.
            </small>
          )}
        </label>

        <label>
          Experience

          <select
            value={setup.experience}
            onChange={(e) =>
              update(
                'experience',
                e.target.value
              )
            }
          >
            {[
              'Fresher',
              '1–2 Years',
              '3–5 Years',
              '5+ Years',
            ].map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>

          <small className="field-hint">
            Choose the experience level you want the interviewer
            to simulate.
          </small>
        </label>

        <label>
          Interview type

          <div className="segmented">
            {[
              'Technical',
              'HR',
              'Behavioral',
              'Coding',
              'SQL',
              'Mixed',
            ].map((x) => (
              <button
                type="button"
                className={
                  setup.interview_type === x
                    ? 'selected'
                    : ''
                }
                onClick={() =>
                  update(
                    'interview_type',
                    x
                  )
                }
                key={x}
              >
                {x}
              </button>
            ))}
          </div>

          {setup.interview_type === 'Mixed' && (
            <small className="field-hint">
              Mixes technical, project, behavioral and role-specific
              questions.
            </small>
          )}

          {setup.interview_type === 'Technical' &&
            missingSkills.length > 0 && (
              <small className="field-hint">
                Adaptive focus: {missingSkills.slice(0, 5).join(', ')}
                {missingSkills.length > 5 ? '…' : ''}
              </small>
            )}
        </label>

        <label>
          Difficulty

          <div className="difficulty-row">
            {[
              'Easy',
              'Medium',
              'Hard',
            ].map((x) => (
              <button
                type="button"
                className={`${setup.difficulty === x ? 'selected' : ''} ${x.toLowerCase()}`}
                onClick={() =>
                  update(
                    'difficulty',
                    x
                  )
                }
                key={x}
              >
                <span />
                {x}
              </button>
            ))}
          </div>
        </label>

        <label>
          Questions

          <div className="question-count">
            {[5, 10, 15, 20].map((x) => (
              <button
                type="button"
                className={
                  setup.total_questions === x
                    ? 'selected'
                    : ''
                }
                onClick={() =>
                  update(
                    'total_questions',
                    x
                  )
                }
                key={x}
              >
                {x}
              </button>
            ))}
          </div>
        </label>

        <label className="video-toggle">
          <span>
            <strong>Video interview</strong>
            <small>
              Use your webcam and microphone while answering.
            </small>
          </span>

          <input
            type="checkbox"
            checked={Boolean(setup.video_mode)}
            onChange={(e) =>
              update(
                'video_mode',
                e.target.checked
              )
            }
          />
        </label>

        {hasProfile && interviewTopics.length > 0 && (
          <div className="setup-topics">
            <div className="eyebrow">
              AI will prioritize
            </div>

            <div className="tag-list">
              {interviewTopics
                .slice(0, 8)
                .map((topic, index) => (
                  <span
                    className="tag"
                    key={`${topic}-${index}`}
                  >
                    {topic}
                  </span>
                ))}
            </div>
          </div>
        )}

        <button
          className="primary-button large"
          disabled={loading}
        >
          {loading
            ? 'Preparing your questions…'
            : 'Enter interview room'}

          <ArrowRight size={17} />
        </button>
      </form>
    </main>
  );
}

function Interview({
  interview,
  answer,
  setAnswer,
  evaluation,
  onSubmit,
  loading,
  llmMode,
  token,
  videoMode,
  onExit,
}) {
  const progress =
    interview.total_questions > 0
      ? (interview.answered /
          interview.total_questions) *
        100
      : 0;

  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [mediaBusy, setMediaBusy] = useState(false);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setElapsedSeconds((value) => value + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  const formatElapsed = (value) =>
    `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;

  const isRealAI =
    llmMode === 'gemini' ||
    llmMode === 'real-llm' ||
    evaluation?.source === 'llm';

  const interviewType = String(
    interview?.interview_type || ''
  ).toLowerCase();

  const isCoding =
    interviewType.includes('coding') ||
    interviewType.includes('programming');

  const isSQL =
    interviewType.includes('sql') ||
    interviewType.includes('database');

  const answerPlaceholder = isCoding
    ? 'Write your solution here. Include clean code and a short explanation if needed.'
    : isSQL
      ? 'Write your SQL query here. Add a short explanation of your approach if needed.'
      : 'Think out loud. A clear, specific answer is more useful than a perfect one.';

  return (
    <main className="interview-page">
      <div className="interview-top">
        <div>
          <div className="eyebrow">
            {interview.role} · {interview.difficulty}
          </div>

          <h1>
            Stay curious. Take your time.
          </h1>
        </div>

        <button
          className="text-button"
          onClick={onExit}
        >
          <X size={16} />
          Leave session
        </button>
      </div>

      <div className="progress-line">
        <span
          style={{
            width: `${progress}%`,
          }}
        />
      </div>

      <div className="interview-meta">
        <span>
          Question {interview.answered + 1}
          <i>
            / {interview.total_questions}
          </i>
        </span>

        <span>
          <Clock3 size={16} />
          {formatElapsed(elapsedSeconds)}
        </span>
      </div>

      <section className="question-card">
        <span className="question-label">
          {isCoding
            ? 'Coding interviewer'
            : isSQL
              ? 'SQL interviewer'
              : interview.question_source === 'question-bank'
                ? 'AI interviewer · fallback question bank'
                : interview.question_source === 'adaptive-planner'
                  ? 'AI interviewer · adaptive planner'
                  : isRealAI
                    ? 'AI interviewer · live model'
                    : 'AI interviewer · local fallback'}
        </span>

        {isCoding && (
          <div className="interview-mode-badge">
            <strong>CODING CHALLENGE</strong>
            <span>Write a solution, then submit it for evaluation.</span>
          </div>
        )}

        {isSQL && (
          <div className="interview-mode-badge">
            <strong>SQL CHALLENGE</strong>
            <span>Write the SQL query, then submit it for evaluation.</span>
          </div>
        )}

        <h2>
          {interview.next_question}
        </h2>

        {videoMode && (
          <VideoRecorder
            interviewId={interview.id}
            questionNumber={interview.answered + 1}
            token={token}
            onTranscript={(text) => {
              setAnswer(text || '');
            }}
            onRecordingChange={setMediaBusy}
            disabled={loading}
          />
        )}

        <textarea
          className={
            isCoding || isSQL
              ? 'answer-editor code-editor'
              : 'answer-editor'
          }
          value={answer}
          onChange={(e) =>
            setAnswer(e.target.value)
          }
          placeholder={answerPlaceholder}
          spellCheck={!isCoding && !isSQL}
          autoCapitalize={isCoding || isSQL ? 'none' : undefined}
          autoCorrect={isCoding || isSQL ? 'off' : undefined}
          disabled={loading}
        />

        <div className="answer-actions">
          <span>
            {
              answer
                .split(/\s+/)
                .filter(Boolean).length
            }{' '}
            words
          </span>

          <button
            className="primary-button"
            onClick={onSubmit}
            disabled={loading}
          >
            {mediaBusy
              ? 'Finish saving your response first'
              : loading
                ? 'AI is analyzing your answer…'
                : 'Submit answer'}

            <ArrowRight size={16} />
          </button>
        </div>
      </section>

      {evaluation && (
        <aside className="evaluation">
          <div>
            <span className="eyebrow">
              {evaluation.source === 'llm'
                ? 'AI evaluation'
                : 'Evaluation notice'}
            </span>

            <h3>
              {evaluation.overall_score}/10
              <small> overall</small>
            </h3>
          </div>

          <p>
            {evaluation.fallback_reason ||
              evaluation.improvement_tips?.[0] ||
              'Answer evaluated successfully.'}
          </p>
        </aside>
      )}
    </main>
  );
}


function VideoRecorder({
  interviewId,
  questionNumber,
  token,
  onTranscript,
  onRecordingChange,
  disabled,
}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const recognitionRef = useRef(null);
  const shouldRecognizeRef = useRef(false);
  const finalTranscriptRef = useRef('');

  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [transcript, setTranscript] = useState('');
  const [speechAvailable, setSpeechAvailable] = useState(false);
  const [speechListening, setSpeechListening] = useState(false);

  useEffect(() => {
    let mounted = true;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });

        if (!mounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraReady(true);
        setMessage('Camera and microphone ready. Start recording when you are ready.');
      } catch {
        setMessage('Camera/microphone permission was blocked. You can still type your answer.');
      }
    };

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    setSpeechAvailable(Boolean(SpeechRecognition));
    if (!SpeechRecognition) {
      setMessage('Speech-to-text is not available in this browser. Chrome or Edge is recommended.');
    }

    if (navigator.mediaDevices?.getUserMedia) {
      startCamera();
    } else {
      setMessage('This browser does not support webcam access. You can still type your answer.');
    }

    return () => {
      mounted = false;
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        try { recorderRef.current.stop(); } catch {}
      }
      shouldRecognizeRef.current = false;
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch {}
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      onRecordingChange?.(false);
    };
  }, [onRecordingChange]);

  useEffect(() => {
    if (!recording) return undefined;
    const timer = window.setInterval(() => {
      setSeconds((value) => value + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [recording]);

  const startSpeech = async (initialTranscript = '') => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setSpeechAvailable(false);
      setSpeechListening(false);
      setMessage('Speech-to-text is not supported in this browser. Use current Google Chrome or Microsoft Edge, or type your answer.');
      return;
    }

    shouldRecognizeRef.current = true;
    finalTranscriptRef.current = initialTranscript || '';
    // Prefer on-device dictation when the browser supports it. This avoids
    // the Chrome/Brave cloud SpeechRecognition "network" error seen on some
    // machines. The language pack is installed once and then recognition can
    // work locally. Older browsers simply fall back to their normal service.
    let useLocal = false;
    try {
      if (typeof SpeechRecognition.available === 'function') {
        let availability = await SpeechRecognition.available({
          langs: ['en-US'],
          processLocally: true,
          quality: 'dictation',
        });

        if (availability === 'downloadable') {
          setMessage('🎤 Preparing offline speech recognition…');
          const installed = await SpeechRecognition.install({
            langs: ['en-US'],
            quality: 'dictation',
          });
          availability = installed ? 'available' : 'unavailable';
        }

        if (availability === 'downloading') {
          setMessage('🎤 Downloading the offline speech model… please wait.');
          await new Promise((resolve) => window.setTimeout(resolve, 1200));
          availability = await SpeechRecognition.available({
            langs: ['en-US'],
            processLocally: true,
            quality: 'dictation',
          });
        }

        useLocal = availability === 'available';
      }
    } catch (error) {
      console.warn('On-device speech setup failed; trying browser speech service.', error);
      useLocal = false;
    }

    if (!shouldRecognizeRef.current) return;

    setSpeechAvailable(true);

    const launchRecognition = () => {
      if (!shouldRecognizeRef.current) return;

      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      // en-US is deliberately used here because it has broader speech-pack
      // support. It still handles Indian English well for normal interview
      // answers and avoids the en-IN availability issue on some browsers.
      recognition.lang = 'en-US';

      if (useLocal && 'processLocally' in recognition) {
        try {
          recognition.processLocally = true;
        } catch {}
      }

      recognition.onstart = () => {
        setSpeechListening(true);
        setMessage(
          useLocal
            ? '🎤 Listening locally — speak your answer. Your words will appear in the answer box.'
            : '🎤 Listening — speak your answer. Your words will appear in the answer box.'
        );
      };

      recognition.onresult = (event) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const piece = event.results[i][0].transcript.trim();
          if (!piece) continue;
          if (event.results[i].isFinal) {
            finalTranscriptRef.current = `${finalTranscriptRef.current} ${piece}`.trim();
          } else {
            interim += `${piece} `;
          }
        }

        const combined = `${finalTranscriptRef.current} ${interim.trim()}`.trim();
        setTranscript(combined);
        onTranscript(combined);
      };

      recognition.onaudiostart = () => {
        setMessage(
          useLocal
            ? '🎤 Microphone active — local speech recognition is listening.'
            : '🎤 Microphone active — speak now. Your words will appear in the answer box.'
        );
      };

      recognition.onerror = async (event) => {
        const code = event?.error || 'unknown';

        if (code === 'not-allowed' || code === 'service-not-allowed') {
          shouldRecognizeRef.current = false;
          setSpeechListening(false);
          setMessage('🎤 Speech permission was blocked. Allow microphone access for this site and try again.');
          return;
        }

        if (code === 'language-not-supported' && useLocal) {
          // A local language pack may have become unavailable. Stop retrying
          // the same local configuration and let the user type instead.
          useLocal = false;
          setMessage('🎤 Local speech pack is unavailable on this browser. You can still type your answer.');
          return;
        }

        if (code === 'no-speech') {
          setMessage('🎤 No speech detected — keep speaking. Listening will continue.');
        } else if (code === 'network' && useLocal) {
          // A local recognizer should not need the network. If the browser
          // nevertheless reports network, do not endlessly hammer the service.
          setMessage('🎤 Local speech recognition could not start. You can type your answer or try Chrome after installing its speech pack.');
        } else {
          setMessage(`🎤 Speech recognition issue (${code}). It will retry while recording.`);
        }
      };

      recognition.onend = () => {
        recognitionRef.current = null;
        setSpeechListening(false);
        if (shouldRecognizeRef.current) {
          window.setTimeout(launchRecognition, 300);
        }
      };

      recognitionRef.current = recognition;
      try {
        recognition.start();
      } catch {
        recognitionRef.current = null;
        if (shouldRecognizeRef.current) {
          window.setTimeout(launchRecognition, 500);
        }
      }
    };

    launchRecognition();
  };

  const stopSpeech = () => {
    shouldRecognizeRef.current = false;
    setSpeechListening(false);
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch {}
      recognitionRef.current = null;
    }
  };

  const startRecording = () => {
    if (!streamRef.current || !window.MediaRecorder) {
      setMessage('Video recording is not supported by this browser. You can still type your answer.');
      return;
    }

    chunksRef.current = [];
    setSeconds(0);
    setTranscript('');
    finalTranscriptRef.current = '';
    setMessage(speechAvailable
      ? 'Recording video + microphone. Your speech will appear in the answer box.'
      : 'Recording video + microphone. Type your answer while recording.'
    );

    const mimeCandidates = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
    ];
    const mimeType =
      mimeCandidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';

    const recorder = new MediaRecorder(
      streamRef.current,
      mimeType ? { mimeType } : undefined
    );

    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.start(1000);
    setRecording(true);
    onRecordingChange?.(true);
    startSpeech('');
  };

  const stopRecording = async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;

    stopSpeech();
    setRecording(false);
    onRecordingChange?.(true);

    const blob = await new Promise((resolve) => {
      recorder.onstop = () => {
        // Store a browser-safe WebM MIME type. Some browsers report
        // codec-specific MIME values that the API may reject.
        resolve(new Blob(chunksRef.current, {
          type: 'video/webm',
        }));
      };
      recorder.stop();
    });

    if (!blob.size) {
      setMessage('No video was captured. Please try again.');
      onRecordingChange?.(false);
      return;
    }

    setUploading(true);
    onRecordingChange?.(true);
    setMessage('Saving your video response…');

    try {
      const form = new FormData();
      form.append('video', blob, `question-${questionNumber}.webm`);

      const response = await fetch(
        `${API}/interviews/${interviewId}/video?question_number=${questionNumber}&duration_seconds=${seconds}`,
        {
          method: 'POST',
          headers: token ? { Authorization: `Bearer ${token}` } : {},
          body: form,
        }
      );

      const body = await response.json();
      if (!response.ok) throw new Error(body?.detail || 'Video upload failed.');
      setMessage('✓ Video response saved. You can submit the answer now.');
    } catch (error) {
      setMessage(error.message || 'Video upload failed. You can still submit your typed answer.');
    } finally {
      setUploading(false);
      onRecordingChange?.(false);
      recorderRef.current = null;
    }
  };

  const formatTime = (value) =>
    `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;

  return (
    <div className="video-interview-panel">
      <div className="video-header">
        <div>
          <span className="question-label">VOICE + VIDEO INTERVIEW</span>
          <strong>{recording ? 'Recording your answer' : uploading ? 'Saving your response' : 'Camera ready'}</strong>
        </div>
        <span className={recording ? 'recording-pill' : 'status-pill'}>
          {recording
            ? `● ${formatTime(seconds)}`
            : uploading
              ? 'Saving…'
              : cameraReady
                ? 'Ready'
                : 'Starting…'}
        </span>
      </div>

      <video
        ref={videoRef}
        className="candidate-video"
        autoPlay
        muted
        playsInline
      />

      <div className="media-status-row">
        <span className={cameraReady ? 'media-status ready' : 'media-status'}>
          ● Camera {cameraReady ? 'ready' : 'waiting'}
        </span>
        <span className={cameraReady ? 'media-status ready' : 'media-status'}>
          ● Microphone {cameraReady ? 'ready' : 'waiting'}
        </span>
        <span className={speechAvailable ? 'media-status ready' : 'media-status'}>
          ● Speech-to-text {speechAvailable ? (speechListening ? 'listening' : 'available') : 'browser unsupported'}
        </span>
      </div>

      <div className="video-actions">
        {!recording ? (
          <button
            type="button"
            className="primary-button"
            onClick={startRecording}
            disabled={disabled || uploading || !cameraReady}
          >
            ● Start voice + video
          </button>
        ) : (
          <button
            type="button"
            className="primary-button danger-button"
            onClick={stopRecording}
            disabled={uploading}
          >
            ■ Stop & save response
          </button>
        )}

        <span className="transcript-badge">
          {transcript
            ? `Live transcript · ${transcript.split(/\s+/).filter(Boolean).length} words`
            : speechAvailable
              ? 'Speak naturally — transcript will appear in the answer box.'
              : 'Type your answer if speech-to-text is unavailable.'}
        </span>
      </div>

      {transcript && (
        <div className="transcript-preview">
          <span>Live transcript</span>

          <p>{transcript}</p>

          <div className="transcript-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={() => onTranscript(transcript)}
              disabled={disabled || uploading}
            >
              Insert transcript into answer
            </button>

            <button
              type="button"
              className="secondary-button"
              onClick={() => {
                setTranscript('');
                finalTranscriptRef.current = '';
                onTranscript('');
                setMessage('Transcript cleared. You can record a new answer.');
              }}
              disabled={disabled || uploading || recording}
            >
              Clear transcript
            </button>
          </div>
        </div>
      )}

      {message && <small className="video-message">{message}</small>}
    </div>
  );
}

function ProfileWorkspace({ token, profile, setProfile, onStartInterview }) {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdTitle, setJdTitle] = useState('Target Job');
  const [jdText, setJdText] = useState('');
  const [loadingResume, setLoadingResume] = useState(false);
  const [loadingJD, setLoadingJD] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const loadLatest = async () => {
    try {
      const [resume, jd] = await Promise.allSettled([
        request('/resume/latest', { token }),
        request('/job-description/latest', { token }),
      ]);
      setProfile({
        resume: resume.status === 'fulfilled' ? resume.value : null,
        jd: jd.status === 'fulfilled' ? jd.value : null,
      });
    } catch {}
  };

  useEffect(() => { loadLatest(); }, [token]);

  const analyzeResume = async () => {
    if (!resumeFile) return setError('Choose a PDF resume first.');
    setLoadingResume(true); setError(''); setMessage('Extracting and analyzing your resume…');
    try {
      const form = new FormData();
      form.append('file', resumeFile);
      const result = await fetch(`${API}/resume/analyze`, {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: form,
      });
      const body = await result.json();
      if (!result.ok) throw new Error(body?.detail || 'Resume analysis failed.');
      setProfile((p) => ({ ...(p || {}), resume: { ...body, analysis: body.analysis } }));
      setMessage(`Resume analyzed using ${body.analysis_mode === 'local' ? 'local evidence-based analysis' : 'Gemini AI'}.`);
    } catch (e) { setError(e.message); setMessage(''); }
    finally { setLoadingResume(false); }
  };

  const analyzeJD = async () => {
    if (jdText.trim().length < 20) return setError('Paste a meaningful job description (at least 20 characters).');
    setLoadingJD(true); setError(''); setMessage('Analyzing the job description and calculating resume match…');
    try {
      const result = await request('/job-description/analyze', {
        method: 'POST',
        token,
        body: JSON.stringify({ title: jdTitle.trim() || 'Target Job', description: jdText.trim() }),
      });
      setProfile((p) => ({ ...(p || {}), jd: result }));
      setMessage(`Job description analyzed using ${result.analysis_mode === 'local' ? 'local evidence-based analysis' : 'Gemini AI'}.`);
    } catch (e) { setError(e.message); setMessage(''); }
    finally { setLoadingJD(false); }
  };

  const resume = profile?.resume?.analysis || {};
  const jd = profile?.jd?.analysis || {};
  const match = profile?.jd?.resume_match || {};
  const list = (value) => Array.isArray(value) ? value : [];

  return (
    <main className="profile-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Candidate intelligence</div>
          <h1>Build your interview profile.</h1>
          <p>Upload your resume and add a target job. The platform uses both to personalize the interview.</p>
        </div>
        <button className="primary-button" onClick={onStartInterview}>
          <Play size={15} /> Start interview <ArrowRight size={15} />
        </button>
      </div>

      {(error || message) && (
        <div className={error ? 'error-banner' : 'mode-banner'}>{error || message}</div>
      )}

      <div className="profile-grid">
        <section className="profile-card">
          <div className="card-title"><FileText size={19} /><h3>Resume Intelligence</h3></div>
          <p className="muted">PDF only. The analyzer extracts skills, projects, education, experience and realistic preparation gaps.</p>
          <label className="upload-box">
            <UploadCloud size={26} />
            <strong>{resumeFile?.name || profile?.resume?.filename || 'Choose your resume PDF'}</strong>
            <small>Drop/select a PDF, then analyze it.</small>
            <input type="file" accept="application/pdf,.pdf" onChange={(e) => setResumeFile(e.target.files?.[0] || null)} hidden />
          </label>
          <button className="primary-button" onClick={analyzeResume} disabled={loadingResume || !resumeFile}>
            {loadingResume ? 'Analyzing resume…' : 'Analyze resume'} <ArrowRight size={15} />
          </button>

          {resume.candidate_summary && <div className="intel-section"><span className="eyebrow">Candidate summary</span><p>{resume.candidate_summary}</p></div>}
          <IntelTags title="Technical skills" items={resume.technical_skills} />
          <IntelTags title="Data & analytics" items={resume.data_analytics_skills} />
          <IntelTags title="Web development" items={resume.web_development_skills} />
          <IntelTags title="Tools & technologies" items={resume.tools_and_technologies} />
          <IntelTags title="Projects" items={resume.projects} />
          <IntelTags title="Experience" items={resume.experience} />
          <IntelTags title="Education" items={resume.education} />
          <IntelTags title="Skill gaps" items={resume.skill_gaps} warning />
          <IntelTags title="Suitable roles" items={resume.suitable_roles} />
          <IntelTags title="Recommended interview topics" items={resume.recommended_interview_topics} />
          <IntelTags title="Recommendations" items={resume.recommendations} />
          {resume.resume_quality_score != null && <ScoreMini label="Resume quality" value={resume.resume_quality_score} />}
        </section>

        <section className="profile-card">
          <div className="card-title"><BriefcaseBusiness size={19} /><h3>Job Description Intelligence</h3></div>
          <label>Job title<input value={jdTitle} onChange={(e) => setJdTitle(e.target.value)} placeholder="AI/ML Engineer" /></label>
          <label>Paste job description<textarea className="jd-textarea" value={jdText} onChange={(e) => setJdText(e.target.value)} placeholder="Paste the complete job description here…" /></label>
          <button className="primary-button" onClick={analyzeJD} disabled={loadingJD}>
            {loadingJD ? 'Analyzing JD…' : 'Analyze job description'} <ArrowRight size={15} />
          </button>

          {jd.job_role && <div className="intel-section"><span className="eyebrow">Detected role</span><h3>{jd.job_role}</h3><p>{jd.job_summary}</p></div>}
          <IntelTags title="Required skills" items={jd.required_skills} good />
          <IntelTags title="Preferred skills" items={jd.preferred_skills} />
          <IntelTags title="Programming languages" items={jd.programming_languages} />
          <IntelTags title="Frameworks & libraries" items={jd.frameworks_and_libraries} />
          <IntelTags title="Databases" items={jd.databases} />
          <IntelTags title="Tools & platforms" items={jd.tools_and_platforms} />
          <IntelTags title="Responsibilities" items={jd.responsibilities} />
          <IntelTags title="Soft skills" items={jd.soft_skills} />
          <IntelTags title="Experience requirements" items={jd.experience_requirements} />
          <IntelTags title="Education requirements" items={jd.education_requirements} />
          <IntelTags title="Interview topics" items={jd.important_interview_topics} />
        </section>
      </div>

      {match?.match_score != null && (
        <section className="match-card">
          <div className="match-score"><span>Resume ↔ JD match</span><strong>{match.match_score}<small>/100</small></strong><div className="score-track"><span style={{width: `${match.match_score}%`}} /></div></div>
          <div className="match-content">
            <div className="eyebrow">Compatibility analysis</div>
            <h2>{match.match_assessment}</h2>
            <div className="match-columns">
              <IntelTags title="Matched" items={match.matched_skills} good />
              <IntelTags title="Partial" items={match.partially_matched_skills} />
              <IntelTags title="Missing / prepare" items={match.missing_skills} warning />
            </div>
            <div className="match-columns">
              <IntelTags title="Matched requirements" items={match.matched_requirements} good />
              <IntelTags title="Preparation recommendations" items={match.preparation_recommendations} warning />
            </div>
          </div>
        </section>
      )}
    </main>
  );
}

function IntelTags({ title, items, warning = false, good = false }) {
  const values = Array.isArray(items) ? items : [];
  if (!values.length) return null;
  return <div className="intel-section"><span className="eyebrow">{title}</span><div className="tag-list">{values.slice(0, 12).map((item, i) => <span className={`tag ${good ? 'good' : warning ? 'warning' : ''}`} key={`${item}-${i}`}>{item}</span>)}</div></div>;
}

function ScoreMini({ label, value }) {
  return <div className="mini-score"><span>{label}</span><strong>{value}/100</strong><div className="score-track"><span style={{width: `${Math.max(0, Math.min(100, value))}%`}} /></div></div>;
}

function Dashboard({
  user,
  token,
  onNew,
  onReport,
  onAIInsights,
}) {
  const [data, setData] = useState({
    items: [],
    stats: {
      average: 0,
      best: 0,
      completed: 0,
    },
  });

  const [loading, setLoading] =
    useState(true);

  useEffect(() => {
    request('/interviews', { token })
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [token]);

  return (
    <main className="dashboard">
      <div className="dashboard-heading">
        <div>
          <div className="eyebrow">
            Your studio · live data
          </div>

          <h1>
            Good to see you,{' '}
            {user?.name?.split(' ')[0] ||
              'there'}
            .
          </h1>

          <p>
            Small, deliberate practice adds up.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={onNew}
        >
          <Play size={15} />
          New interview
        </button>
      </div>

      <div className="stats">
        <Stat
          label="Average score"
          value={data.stats.average}
          suffix="/100"
        />

        <Stat
          label="Best score"
          value={data.stats.best}
          suffix="/100"
        />

        <Stat
          label="Completed"
          value={data.stats.completed}
          suffix="sessions"
        />
      </div>

      <section className="recent">
        <div className="section-head">
          <div>
            <div className="eyebrow">
              Your practice history
            </div>

            <h2>
              Recent interviews
            </h2>
          </div>

          <span className="live-badge">
            <span />
            Synced
          </span>
        </div>

        <div className="interview-list">
          {loading ? (
            <div className="empty-state">
              Loading your interview history…
            </div>
          ) : data.items.length === 0 ? (
            <div className="empty-state">
              <BarChart3 size={24} />

              <strong>
                Your first session starts here.
              </strong>

              <span>
                Complete an interview to see your
                progress and feedback.
              </span>

              <button
                className="primary-button"
                onClick={onNew}
              >
                Start interview
                <ArrowRight size={15} />
              </button>
            </div>
          ) : (
            data.items.map((item) => (
              <div
                className="interview-row"
                key={item.id}
              >
                <div className="row-icon">
                  <BarChart3 size={18} />
                </div>

                <div className="row-main">
                  <strong>
                    {item.role}
                  </strong>

                  <span>
                    {item.interview_type} ·{' '}
                    {item.answered}/
                    {item.total_questions}{' '}
                    questions ·{' '}
                    {new Date(
                      item.created_at
                    ).toLocaleDateString()}
                  </span>
                </div>

                <div
                  className={`status-pill ${item.status}`}
                >
                  {item.status}
                </div>

                <div className="row-score">
                  <strong>
                    {item.score ?? '—'}
                  </strong>

                  <span>
                    {item.score === null
                      ? 'active'
                      : '/100'}
                  </span>
                </div>

                <div className="dashboard-actions">
                  <button
                    className="row-button"
                    onClick={() =>
                      item.status === 'completed'
                        ? onReport(item.id)
                        : onNew()
                    }
                  >
                    {item.status === 'completed' ? 'View report' : 'Resume'}
                    <ArrowRight size={14} />
                  </button>
                  {item.status === 'completed' && (
                    <button
                      className="row-button"
                      onClick={() => onAIInsights(item.id)}
                    >
                      <Activity size={14} />
                      AI Insights
                    </button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </main>
  );
}

function Stat({
  label,
  value,
  suffix,
}) {
  return (
    <div className="stat">
      <span>{label}</span>

      <strong>
        {value}
        <small>{suffix}</small>
      </strong>
    </div>
  );
}

/* =========================================================
   REPORT
   ========================================================= */

function AIInsights({ assessment, onBack }) {
  if (!assessment) {
    return (
      <main className="center-page">
        <div className="auth-card">
          <h1>No AI assessment yet.</h1>
          <p>Complete an interview and open AI Insights from the dashboard.</p>
          <button className="primary-button" onClick={onBack}>Back to dashboard</button>
        </div>
      </main>
    );
  }

  const readiness = assessment.readiness || {};
  const dimensions = assessment.dimension_scores || {};

  return (
    <main className="report-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Explainable AI / ML assessment</div>
          <h1>AI interview intelligence.</h1>
          <p>NLP signals and a machine-learning readiness prediction, alongside the qualitative LLM review.</p>
        </div>
        <button className="secondary-button" onClick={onBack}>Back to dashboard</button>
      </div>

      <section className="ai-hero-grid">
        <article className="ai-score-card">
          <div className="eyebrow">Readiness prediction</div>
          <strong>{readiness.score ?? 0}<span>/100</span></strong>
          <h3>{readiness.label || 'Not enough data'}</h3>
          <p>{readiness.trend > 0 ? `+${readiness.trend}` : readiness.trend} trend across this session.</p>
        </article>
        <article className="ai-method-card">
          <div className="eyebrow"><Brain size={15} /> Model</div>
          <h3>{assessment.model}</h3>
          <p>Features combine answer evaluation, semantic similarity, concept coverage, answer length and filler-word ratio.</p>
          <div className="mini-metrics">
            <span>MAE <b>{readiness.metrics?.mae ?? '—'}</b></span>
            <span>R² <b>{readiness.metrics?.r2 ?? '—'}</b></span>
          </div>
        </article>
      </section>

      <section className="analytics-grid">
        <article className="report-card">
          <div className="eyebrow">Performance signals</div>
          <h2>Dimension profile</h2>
          {Object.entries(dimensions).map(([name, score]) => (
            <div className="signal-row" key={name}>
              <div><span>{name}</span><b>{score}</b></div>
              <div className="signal-track"><i style={{ width: `${Math.max(0, Math.min(100, score))}%` }} /></div>
            </div>
          ))}
        </article>

        <article className="report-card">
          <div className="eyebrow"><Mic2 size={15} /> NLP / communication</div>
          <h2>What the model sees</h2>
          <div className="metric-list">
            <div><span>Semantic similarity</span><b>{readiness.average_semantic_similarity ?? 0}</b></div>
            <div><span>Concept coverage</span><b>{readiness.average_concept_coverage ?? 0}%</b></div>
            <div><span>Filler ratio</span><b>{readiness.average_filler_ratio ?? 0}</b></div>
          </div>
        </article>
      </section>

      <section className="report-card">
        <div className="eyebrow">Personalized next steps</div>
        <h2>Priority areas</h2>
        <div className="tag-list">
          {(assessment.priority_areas || []).map((x) => <span key={x}>{x}</span>)}
        </div>
        <ul className="plan-list">
          {(assessment.recommendations || []).map((x, i) => <li key={i}>{x}</li>)}
        </ul>
      </section>

      <section className="report-card methodology-card">
        <div className="eyebrow">Research transparency</div>
        <h2>Methodology</h2>
        <p><strong>NLP:</strong> {(assessment.methodology?.nlp || []).join(', ')}.</p>
        <p><strong>ML:</strong> {assessment.methodology?.ml}.</p>
        <p className="muted">{assessment.methodology?.note}</p>
      </section>

      <section className="report-card">
        <div className="eyebrow">Answer-level analysis</div>
        <h2>Signals by question</h2>
        <div className="answer-signal-grid">
          {(assessment.answer_analytics || []).map((a, i) => (
            <article key={i} className="answer-signal-card">
              <strong>Q{i + 1}</strong>
              <span>{a.readiness_prediction}/100 predicted readiness</span>
              <small>{a.word_count} words · {a.filler_count} fillers · {a.concept_coverage}% concept coverage · {a.semantic_similarity} semantic similarity</small>
              {a.keyword_hits?.length > 0 && <small>Detected concepts: {a.keyword_hits.join(', ')}</small>}
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}

function Report({
  report,
  onNew,
}) {
  const [expandedReview, setExpandedReview] =
    useState(null);

  if (!report) {
    return (
      <main className="center-page">
        <div className="auth-card">
          <div className="eyebrow">
            Report unavailable
          </div>

          <h1>
            No report loaded.
          </h1>

          <button
            className="primary-button"
            onClick={onNew}
          >
            Start a new interview
            <ArrowRight size={16} />
          </button>
        </div>
      </main>
    );
  }

  const overallScore =
    Number(report.overall_score) || 0;

  const dimensionScores =
    report.dimension_scores || {};

  const strongAreas =
    report.strong_areas ||
    report.strongest_skills ||
    [];

  const weakAreas =
    report.weak_areas ||
    report.skill_gaps ||
    [];

  const improvementPlan =
    report.improvement_plan || [];

  const sevenDayPlan =
    report.seven_day_plan || [];

  const aiAssessment =
    report.ai_assessment || {};

  const readiness =
    report.readiness ||
    aiAssessment.role_readiness ||
    'Not available';

  const summary =
    aiAssessment.summary ||
    report.summary ||
    'Your interview has been evaluated. Review the areas below to understand where you performed well and where your next practice session should focus.';

  const recommendation =
    report.recommendation ||
    'Continue practicing the areas identified in your skill gaps.';

  const dimensionLabels = [
    ['Correctness', 'correctness'],
    ['Technical knowledge', 'technical_knowledge'],
    ['Relevance', 'relevance'],
    ['Clarity', 'clarity'],
    ['Communication', 'communication'],
  ];

  return (
    <main className="report-page">
      {/* HEADER */}

      <div className="report-heading">
        <div>
          <div className="eyebrow">
            Session complete · detailed readout
          </div>

          <h1>
            Now you know more.
          </h1>

          <p>
            {report.interview?.role ||
              'Interview'}{' '}
            ·{' '}
            {report.interview
              ?.interview_type ||
              'Mixed'}{' '}
            ·{' '}
            {report.reviews?.length || 0}{' '}
            answers reviewed
          </p>
        </div>

        <button
          className="primary-button"
          onClick={onNew}
        >
          <Play size={15} />
          Practice again
        </button>
      </div>

      {/* TOP SCORE SECTION */}

      <section className="report-overview">
        <div className="score-card report-score-card">
          <div className="report-score-top">
            <div>
              <span>Overall score</span>

              <strong>
                {overallScore}
                <small>/100</small>
              </strong>
            </div>

            <div className="score-icon">
              <Award size={24} />
            </div>
          </div>

          <div className="score-track">
            <span
              style={{
                width: `${overallScore}%`,
              }}
            />
          </div>

          <div className="score-footer">
            <span>
              {getScoreLabel(overallScore)}
            </span>

            <span>
              {report.questions_reviewed ||
                report.reviews?.length ||
                0}{' '}
              questions reviewed
            </span>
          </div>
        </div>

        <div className="readiness-card">
          <div className="card-title">
            <TrendingUp size={18} />

            <h3>
              Role readiness
            </h3>
          </div>

          <strong>
            {readiness}
          </strong>

          <p>
            This assessment reflects your
            performance across the completed
            interview.
          </p>
        </div>
      </section>

      {/* AI SUMMARY */}

      <section className="ai-summary-card">
        <div className="ai-summary-icon">
          <Sparkles size={20} />
        </div>

        <div>
          <div className="eyebrow">
            AI interviewer assessment
          </div>

          <h2>
            Overall assessment
          </h2>

          <p>
            {summary}
          </p>
        </div>
      </section>

      {/* DIMENSION SCORES */}

      <section className="report-section">
        <div className="section-head">
          <div>
            <div className="eyebrow">
              Performance breakdown
            </div>

            <h2>
              Your scoring profile
            </h2>
          </div>

          <span className="review-count">
            5 dimensions
          </span>
        </div>

        <div className="dimension-grid">
          {dimensionLabels.map(
            ([label, key]) => {
              const rawValue =
                Number(
                  dimensionScores[key]
                ) || 0;
              // Backend dimension scores are 0–10; the report UI displays 0–100.
              const value = Math.max(
                0,
                Math.min(100, rawValue * 10)
              );

              return (
                <div
                  className="dimension-card"
                  key={key}
                >
                  <div className="dimension-top">
                    <span>
                      {label}
                    </span>

                    <strong>
                      {Number.isInteger(value)
                        ? value
                        : value.toFixed(1)}
                      <small>/100</small>
                    </strong>
                  </div>

                  <div className="dimension-track">
                    <span
                      style={{
                        width: `${value}%`,
                      }}
                    />
                  </div>
                </div>
              );
            }
          )}
        </div>
      </section>

      {/* STRONG + WEAK */}

      <div className="report-grid">
        <div className="report-card">
          <div className="card-title">
            <CheckCircle2 size={18} />

            <h3>
              Strong areas
            </h3>
          </div>

          {strongAreas.length > 0 ? (
            <div className="tag-list">
              {strongAreas.map(
                (item, index) => (
                  <div
                    className="tag good"
                    key={`${item}-${index}`}
                  >
                    <CheckCircle2 size={14} />
                    {item}
                  </div>
                )
              )}
            </div>
          ) : (
            <p className="muted">
              No strong areas were identified
              yet.
            </p>
          )}
        </div>

        <div className="report-card">
          <div className="card-title">
            <AlertCircle size={18} />

            <h3>
              Skill gaps
            </h3>
          </div>

          {weakAreas.length > 0 ? (
            <div className="tag-list">
              {weakAreas.map(
                (item, index) => (
                  <div
                    className="tag warning"
                    key={`${item}-${index}`}
                  >
                    <AlertCircle size={14} />
                    {item}
                  </div>
                )
              )}
            </div>
          ) : (
            <p className="muted">
              No major skill gaps were
              identified.
            </p>
          )}
        </div>
      </div>

      {/* AI ASSESSMENTS */}

      <section className="assessment-section">
        <div className="section-head">
          <div>
            <div className="eyebrow">
              What the interviewer noticed
            </div>

            <h2>
              AI assessment
            </h2>
          </div>
        </div>

        <div className="assessment-grid">
          <AssessmentCard
            icon={<Code2 size={19} />}
            title="Technical assessment"
            text={
              aiAssessment
                .technical_assessment
            }
          />

          <AssessmentCard
            icon={
              <MessageSquare size={19} />
            }
            title="Communication assessment"
            text={
              aiAssessment
                .communication_assessment
            }
          />

          <AssessmentCard
            icon={
              <BrainCircuit size={19} />
            }
            title="Problem-solving assessment"
            text={
              aiAssessment
                .problem_solving_assessment
            }
          />
        </div>
      </section>

      {/* IMPROVEMENT PLAN */}

      <section className="plan-section">
        <div className="section-head">
          <div>
            <div className="eyebrow">
              Personalized preparation
            </div>

            <h2>
              Your improvement plan
            </h2>
          </div>
        </div>

        {improvementPlan.length > 0 ? (
          <div className="plan-list">
            {improvementPlan.map(
              (item, index) => (
                <div
                  className="plan-item"
                  key={`${item}-${index}`}
                >
                  <div className="plan-number">
                    {String(index + 1).padStart(
                      2,
                      '0'
                    )}
                  </div>

                  <div>
                    <strong>
                      Focus area {index + 1}
                    </strong>

                    <p>{item}</p>
                  </div>
                </div>
              )
            )}
          </div>
        ) : (
          <div className="empty-plan">
            <Lightbulb size={20} />
            <span>
              Complete more interviews to
              generate a more detailed
              improvement plan.
            </span>
          </div>
        )}
      </section>

      {/* 7 DAY PLAN */}

      <section className="seven-day-section">
        <div className="section-head">
          <div>
            <div className="eyebrow">
              Preparation roadmap
            </div>

            <h2>
              Your 7-day plan
            </h2>
          </div>

          <CalendarDays size={21} />
        </div>

        {sevenDayPlan.length > 0 ? (
          <div className="seven-day-grid">
            {sevenDayPlan.map(
              (item, index) => (
                <div
                  className="day-card"
                  key={`${item}-${index}`}
                >
                  <span>
                    Day {index + 1}
                  </span>

                  <p>{item}</p>
                </div>
              )
            )}
          </div>
        ) : (
          <div className="empty-plan">
            <CalendarDays size={20} />

            <span>
              Your 7-day preparation roadmap
              will appear here after AI report
              generation.
            </span>
          </div>
        )}
      </section>

      {/* RECOMMENDATION */}

      <section className="recommendation">
        <span className="eyebrow">
          A note from your interviewer
        </span>

        <h2>
          {recommendation}
        </h2>

        <p>
          Return to the topics above in your
          next practice session. The goal is
          progress you can explain.
        </p>
      </section>

      {/* QUESTION REVIEW */}

      <section className="review-section">
        <div className="section-head">
          <div>
            <div className="eyebrow">
              Answer by answer
            </div>

            <h2>
              Session review
            </h2>
          </div>

          <span className="review-count">
            {report.reviews?.length || 0}{' '}
            reviewed
          </span>
        </div>

        {(report.reviews || []).map(
          (review, index) => {
            const isExpanded =
              expandedReview === index;

            return (
              <article
                className="review-card"
                key={`${review.question}-${index}`}
              >
                <button
                  className="review-card-header"
                  onClick={() =>
                    setExpandedReview(
                      isExpanded
                        ? null
                        : index
                    )
                  }
                >
                  <div className="review-number">
                    {String(
                      index + 1
                    ).padStart(2, '0')}
                  </div>

                  <div className="review-header-main">
                    <strong>
                      {review.question}
                    </strong>

                    <span>
                      Score:{' '}
                      {review.evaluation
                        ?.overall_score ??
                        0}
                      /10
                    </span>
                  </div>

                  <div className="review-score">
                    {review.evaluation
                      ?.overall_score ??
                      0}
                    /10
                  </div>

                  <div className="review-expand">
                    {isExpanded ? (
                      <ChevronUp size={18} />
                    ) : (
                      <ChevronDown size={18} />
                    )}
                  </div>
                </button>

                {isExpanded && (
                  <div className="review-card-body">
                    <div className="candidate-answer-box">
                      <span>
                        Candidate answer
                      </span>

                      <p>
                        “
                        {review.answer ||
                          'No answer recorded.'}
                        ”
                      </p>
                    </div>

                    <div className="review-metrics">
                      {[
                        [
                          'Correctness',
                          'correctness',
                        ],
                        [
                          'Technical',
                          'technical_knowledge',
                        ],
                        [
                          'Relevance',
                          'relevance',
                        ],
                        [
                          'Clarity',
                          'clarity',
                        ],
                        [
                          'Communication',
                          'communication',
                        ],
                      ].map(
                        ([label, key]) => (
                          <span key={key}>
                            <small>
                              {label}
                            </small>

                            <b>
                              {review
                                .evaluation?.[
                                key
                              ] ?? 0}
                              /10
                            </b>
                          </span>
                        )
                      )}
                    </div>

                    {review.evaluation
                      ?.strengths
                      ?.length > 0 && (
                      <div className="review-detail">
                        <strong>
                          Strengths
                        </strong>

                        <ul>
                          {review.evaluation.strengths.map(
                            (
                              item,
                              strengthIndex
                            ) => (
                              <li
                                key={
                                  strengthIndex
                                }
                              >
                                {item}
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                    {review.evaluation
                      ?.weaknesses
                      ?.length > 0 && (
                      <div className="review-detail">
                        <strong>
                          Weaknesses
                        </strong>

                        <ul>
                          {review.evaluation.weaknesses.map(
                            (
                              item,
                              weaknessIndex
                            ) => (
                              <li
                                key={
                                  weaknessIndex
                                }
                              >
                                {item}
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                    {review.evaluation
                      ?.missing_points
                      ?.length > 0 && (
                      <div className="review-detail">
                        <strong>
                          Missing points
                        </strong>

                        <ul>
                          {review.evaluation.missing_points.map(
                            (
                              item,
                              missingIndex
                            ) => (
                              <li
                                key={
                                  missingIndex
                                }
                              >
                                {item}
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                    {review.evaluation
                      ?.improvement_tips
                      ?.length > 0 && (
                      <div className="review-detail tip">
                        <strong>
                          How to improve
                        </strong>

                        <ul>
                          {review.evaluation.improvement_tips.map(
                            (
                              item,
                              tipIndex
                            ) => (
                              <li
                                key={tipIndex}
                              >
                                {item}
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                    {review.evaluation
                      ?.ideal_answer && (
                      <div className="ideal-answer">
                        <strong>
                          Ideal answer
                        </strong>

                        <p>
                          {
                            review
                              .evaluation
                              .ideal_answer
                          }
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </article>
            );
          }
        )}
      </section>
    </main>
  );
}

function AssessmentCard({
  icon,
  title,
  text,
}) {
  return (
    <article className="assessment-card">
      <div className="assessment-icon">
        {icon}
      </div>

      <h3>{title}</h3>

      <p>
        {text ||
          'The AI interviewer did not provide a separate assessment for this category.'}
      </p>
    </article>
  );
}

function getScoreLabel(score) {
  if (score >= 85) {
    return 'Excellent performance';
  }

  if (score >= 70) {
    return 'Strong performance';
  }

  if (score >= 55) {
    return 'Developing performance';
  }

  if (score >= 40) {
    return 'Needs focused practice';
  }

  return 'Needs significant improvement';
}

createRoot(
  document.getElementById('root')
).render(<App />);