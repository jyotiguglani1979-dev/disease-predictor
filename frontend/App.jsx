import React, { useEffect, useRef, useState } from 'react';
import {
  Activity, ArrowRight, BadgeInfo, BookOpen, CalendarDays, Check,
  ChevronRight, CircleHelp, History, Home, MessageCircle, ShieldAlert,
  ShieldCheck, Sparkles, X,
} from 'lucide-react';

const API = 'http://localhost:8000/api';
let guestSessionRequest;

async function guestToken() {
  let token = localStorage.getItem('wellwise_token');
  if (token) return token;
  if (!guestSessionRequest) {
    guestSessionRequest = fetch(`${API}/session/anonymous`, { method: 'POST' })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Could not start a local session');
        localStorage.setItem('wellwise_token', data.token);
        return data.token;
      })
      .finally(() => { guestSessionRequest = null; });
  }
  return guestSessionRequest;
}

async function call(path, options = {}) {
  const needsSession = path === '/history'
    || path === '/assessment/start'
    || /^\/assessment\/[^/]+\/(answer|explain)$/.test(path);
  const token = needsSession ? await guestToken() : localStorage.getItem('wellwise_token');
  const response = await fetch(API + path, {
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  let data;
  try {
    data = await response.json();
  } catch {
    data = { detail: 'Unexpected server response' };
  }
  if (response.status === 401 && token) {
    localStorage.removeItem('wellwise_token');
    window.location.reload();
  }
  if (!response.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

const screens = [
  ['Overview', Home],
  ['Symptom check', Activity],
  ['AI assistant', MessageCircle],
  ['History', History],
  ['About', BookOpen],
];

function App() {
  const [screen, setScreen] = useState('Overview');
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState('');

  useEffect(() => {
    call('/health').then(setStatus).catch(() => setStatus({ status: 'offline' }));
  }, []);

  async function refreshHistory() {
    try {
      setHistory(await call('/history'));
    } catch {}
  }

  useEffect(() => { refreshHistory(); }, []);

  const notify = (message) => {
    setToast(message);
    setTimeout(() => setToast(''), 5000);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-icon"><Activity size={19} /></div><span>wellwise</span><i>HEALTH</i></div>
        <div className="workspace-label">COLLEGE PROJECT</div>
        <nav>
          {screens.map(([name, Icon]) => (
            <button key={name} className={'nav-item ' + (screen === name ? 'active' : '')} onClick={() => setScreen(name)}>
              <Icon size={18} /><span>{name}</span>
              {name === 'History' && history.length > 0 && <small>{history.length}</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="secure"><ShieldCheck size={16} /><span>Educational project · local session</span></div>
        </div>
      </aside>
      <main className="main">
        <header>
          <div className="crumb">Your health space <ChevronRight size={14} /><b>{screen}</b></div>
          <div className="header-right">
            <span className="local-status"><i className={status?.status === 'offline' ? 'off' : ''} />
              {status?.status === 'offline' ? 'API offline' : 'Private session'}
            </span>
          </div>
        </header>
        <div className="content">
          {screen === 'Overview' && <Dashboard go={setScreen} history={history} />}
          {screen === 'Symptom check' && <Symptom status={status} notify={notify} refresh={refreshHistory} />}
          {screen === 'AI assistant' && <Assistant />}
          {screen === 'History' && <HistoryPage rows={history} onClear={refreshHistory} />}
          {screen === 'About' && <About />}
        </div>
      </main>
      {toast && <div className="toast"><BadgeInfo size={17} />{toast}<button onClick={() => setToast('')}><X size={15} /></button></div>}
    </div>
  );
}

function Dashboard({ go, history }) {
  return (
    <>
      <div className="greeting">
        <div>
          <p className="eyebrow">COLLEGE PROJECT</p>
          <h1>A clearer picture of<br /><span>your health.</span></h1>
          <p className="subhead">Organize symptoms and prepare a focused conversation with a licensed clinician.</p>
        </div>
      </div>
      <div className="section-title"><div><p className="eyebrow">YOUR TOOLS</p><h2>Where would you like to begin?</h2></div></div>
      <div className="tool-grid">
        <button className="tool-card symptom-card" onClick={() => go('Symptom check')}>
          <div className="tool-top"><span className="tool-icon green"><Activity size={20} /></span><span className="tool-arrow"><ArrowRight size={17} /></span></div>
          <p className="card-kicker">SYMPTOM ASSESSMENT</p>
          <h3>Compare both<br />model perspectives</h3>
          <p>Review DDXPlus and a 30-disease binary symptom model separately.</p>
          <div className="card-link">Start symptom check <ArrowRight size={15} /></div>
        </button>
        <button className="tool-card assistant-card" onClick={() => go('AI assistant')}>
          <div className="tool-top"><span className="tool-icon blue"><Sparkles size={20} /></span><span className="tool-arrow"><ArrowRight size={17} /></span></div>
          <p className="card-kicker">HEALTH ASSISTANT</p>
          <h3>Get grounded<br />health information</h3>
          <p>Ask general questions without a diagnosis or treatment recommendation.</p>
          <div className="card-link">Ask a question <ArrowRight size={15} /></div>
        </button>
      </div>
      <div className="bottom-grid">
        <div className="recent-panel">
          <div className="panel-heading"><div><p className="eyebrow">YOUR ACTIVITY</p><h3>Recent assessments</h3></div><button className="text-button" onClick={() => go('History')}>View history <ArrowRight size={14} /></button></div>
          {history.length ? (
            <div className="recent-list">{history.slice(0, 3).map((item) => (
              <div className="recent-row" key={item.id}>
                <span className="round-icon green"><Activity size={16} /></span>
                <div><b>Symptom assessment</b><small>{new Date(item.created_at).toLocaleDateString()}</small></div>
                <ChevronRight size={16} />
              </div>
            ))}</div>
          ) : <div className="empty-recent"><CalendarDays size={19} /><span>Your completed assessments will appear here.</span></div>}
        </div>
        <div className="care-note"><div className="note-icon"><ShieldAlert size={18} /></div><div><b>Need urgent help?</b><p>Call local emergency services for severe breathing trouble, chest pain, stroke signs, fainting, or rapid deterioration.</p></div></div>
      </div>
    </>
  );
}

function Symptom({ status, notify, refresh }) {
  const [text, setText] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('Female');
  const [consent, setConsent] = useState(false);
  const [allowReview, setAllowReview] = useState(false);
  const [urgent, setUrgent] = useState('');
  const [record, setRecord] = useState(null);
  const [question, setQuestion] = useState(null);
  const [reviewStatus, setReviewStatus] = useState('idle');
  const [reviewError, setReviewError] = useState('');
  const [finalShortlist, setFinalShortlist] = useState([]);
  const [loading, setLoading] = useState(false);
  const [answers, setAnswers] = useState(0);
  const reviewRequestId = useRef(0);

  async function requestReview(assessmentId) {
    const requestId = ++reviewRequestId.current;
    setFinalShortlist([]);
    setReviewError('');
    if (!allowReview) {
      setReviewStatus('consent');
      return;
    }
    setReviewStatus('loading');
    try {
      const response = await call(`/assessment/${assessmentId}/explain`, {
        method: 'POST',
        body: JSON.stringify({ allow_gemini: true, free_text: text }),
      });
      if (requestId === reviewRequestId.current) {
        setFinalShortlist((response.final_shortlist || []).slice(0, 5));
        setReviewStatus('ready');
      }
    } catch (error) {
      if (requestId === reviewRequestId.current) {
        setReviewError(error.message || 'The optional review could not complete.');
        setReviewStatus('unavailable');
      }
    }
  }

  async function start() {
    if (!consent) {
      notify('Consent is required before this assessment is saved.');
      return;
    }
    if (!age || !text.trim()) {
      notify('Enter your age and describe at least one symptom.');
      return;
    }
    setLoading(true);
    try {
      const response = await call('/assessment/start', {
        method: 'POST',
        body: JSON.stringify({ age: Number(age), sex, evidence: [], free_text: text, consent, allow_gemini: allowReview }),
      });
      if (response.emergency_warning) {
        setUrgent(response.emergency_warning);
        return;
      }
      setRecord(response.assessment);
      setQuestion(response.next_question);
      setAnswers(0);
      refresh();
      if (!allowReview) setReviewStatus('consent');
      else if (response.next_question) setReviewStatus('pending');
      else requestReview(response.assessment.id);
    } catch (error) {
      notify(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function respond(code, token, response) {
    setLoading(true);
    try {
      const updated = await call(`/assessment/${record.id}/answer`, {
        method: 'POST',
        body: JSON.stringify({ code, evidence: token || null, response, allow_gemini: allowReview }),
      });
      setRecord(updated.assessment);
      setQuestion(updated.next_question);
      setAnswers((value) => value + 1);
      refresh();
      if (!allowReview) setReviewStatus('consent');
      else if (updated.next_question) setReviewStatus('pending');
      else requestReview(updated.assessment.id);
    } catch (error) {
      notify(error.message);
    } finally {
      setLoading(false);
    }
  }

  function addSelectedValue() {
    const value = document.getElementById('follow-value')?.value;
    if (!value) {
      notify('Choose a response first.');
      return;
    }
    respond(question.code, `${question.code}_@_${value}`, 'yes');
  }

  return (
    <>
      <div className="page-heading">
        <div><p className="eyebrow">SYMPTOM ASSESSMENT</p><h1>How do you feel?</h1><p className="subhead">Describe symptoms and answer optional follow-up questions.</p></div>
        <div className="dataset-badge"><span className="status-dot" />{status?.indian_model_ready ? 'Models ready' : 'Checking models'}</div>
      </div>
      {urgent && <div className="urgent-banner"><ShieldAlert size={20} /><span>{urgent}</span></div>}
      {!record ? (
        <div className="assessment-layout intake-only">
          <div className="form-panel">
            <div className="step-header"><span>01</span><div><h2>Describe what you’re experiencing</h2><p>Include duration, temperature, pain, and digestive or breathing changes.</p></div></div>
            <label className="field-label" htmlFor="story">Your symptoms</label>
            <textarea id="story" value={text} onChange={(event) => { setText(event.target.value); setUrgent(''); }} placeholder="Example: Rising fever for 3 days with weakness, stomach pain, headache, and loose stools." />
            <div className="form-row" style={{ marginTop: 16 }}>
              <div><label className="field-label" htmlFor="age">Age</label><input id="age" className="text-input" type="number" min="0" max="120" value={age} onChange={(event) => setAge(event.target.value)} /></div>
              <div><label className="field-label" htmlFor="sex">Sex for model context</label><select id="sex" className="text-input" value={sex} onChange={(event) => setSex(event.target.value)}><option>Female</option><option>Male</option><option>Other / not specified</option></select></div>
            </div>
            <label className="consent optional-consent"><input type="checkbox" checked={allowReview} onChange={(event) => setAllowReview(event.target.checked)} /><span>Use optional AI-assisted final review.</span></label>
            <label className="consent"><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} /><span>I consent to saving this educational assessment.</span></label>
            <button className="primary-button full" onClick={start} disabled={loading || !status?.symptom_model_ready || !status?.indian_model_ready || !text.trim()}>
              {loading ? 'Analyzing…' : 'Analyze symptoms'}<ArrowRight size={17} />
            </button>
          </div>
        </div>
      ) : (
        <div className="results-layout">
          <div className="result-main">
            <div className="result-header">
              <div className="result-check"><Check size={20} /></div>
              <div><p className="eyebrow">ASSESSMENT UPDATED · {answers} FOLLOW-UP{answers === 1 ? '' : 'S'}</p><h2>Your assessment</h2><p>{record.result.explanation}</p></div>
            </div>

            {record.result.combined?.length > 0 && (
              <div className="scores-card">
                <div className="panel-heading"><div><p className="eyebrow">COMBINED RANKING</p><h3>Top possibilities from both models</h3></div></div>
                {record.result.combined.map((item, index) => (
                  <div className="score-row" key={item.condition}>
                    <div className="score-rank">0{index + 1}</div>
                    <div className="score-info">
                      <b>{item.condition}</b>{item.also_called && <small> (also called: {item.also_called})</small>}
                      <div className="score-track"><span style={{ width: `${item.score}%` }} /></div>
                      <small>{item.models_agree ? 'Both models agree on this' : `${item.sources[0]} only`}</small>
                    </div>
                    <strong>{item.score}%<small> mixed score</small></strong>
                  </div>
                ))}
                {record.result.combined_note && <p className="scores-foot">{record.result.combined_note}</p>}
              </div>
            )}

            <details className="model-details" style={{ marginTop: 14 }} open={!(record.result.combined?.length > 0)}>
            <summary style={{ cursor: 'pointer', fontWeight: 600, margin: '8px 0' }}>See each model separately</summary>
            <div className="scores-card">
              <div className="panel-heading"><div><p className="eyebrow">DDXPLUS MODEL</p><h3>Five ranked possibilities</h3></div></div>
              {record.result.results.length ? record.result.results.map((item, index) => (
                <div className="score-row" key={item.condition}>
                  <div className="score-rank">0{index + 1}</div>
                  <div className="score-info"><b>{item.condition}</b><div className="score-track"><span style={{ width: `${item.score}%` }} /></div></div>
                  <strong>{item.score}<small> relative score</small></strong>
                </div>
              )) : <p className="scores-foot">No DDXPlus ranking is available for these findings.</p>}
            </div>

            {record.result.indian_results?.ready && (
              <div className="scores-card" style={{ marginTop: 14 }}>
                <div className="panel-heading"><div><p className="eyebrow">INDIAN BINARY MODEL</p><h3>Thirty-condition binary ranking</h3></div></div>
                {record.result.indian_results.results.length === 0 && (
                  <p className="scores-foot">No ranking: none of the symptoms you described are in this model's 66-symptom list.</p>
                )}
                {record.result.indian_results.results.map((item, index) => (
                  <div className="score-row" key={item.condition}>
                    <div className="score-rank">0{index + 1}</div>
                    <div className="score-info"><b>{item.condition}</b><div className="score-track"><span style={{ width: `${item.score}%` }} /></div></div>
                    <strong>{item.score}%<small> model score</small></strong>
                  </div>
                ))}
                {record.result.indian_results.not_modelled_symptoms?.length > 0 && (
                  <p className="scores-foot">Not used by this model (no matching symptom): {record.result.indian_results.not_modelled_symptoms.join(', ')}</p>
                )}
              </div>
            )}
            </details>

            {reviewStatus === 'loading' && <div className="ai-explanation"><b>Preparing final review…</b></div>}
            {reviewStatus === 'unavailable' && <div className="ai-explanation"><b>Optional review unavailable</b><p>{reviewError}</p></div>}
            {reviewStatus === 'ready' && finalShortlist.length > 0 && (
              <div className="ai-explanation">
                <div className="tool-icon blue"><Sparkles size={16} /></div>
                <div><b>Final shortlist</b>
                  {finalShortlist.map((item, index) => (
                    <p key={item.condition + index}><strong>{index + 1}. {item.condition}</strong><br /><small>{item.reason}</small>{item.source_url && <><br /><a href={item.source_url} target="_blank" rel="noreferrer">{item.source_title}</a></>}</p>
                  ))}
                  <small>This is an educational review, not a diagnosis or laboratory confirmation.</small>
                </div>
              </div>
            )}

            <div className="assessment-actions">
              <button className="secondary-button" onClick={() => { setRecord(null); setQuestion(null); }}>Edit symptoms</button>
              <button className="text-button" onClick={() => { setRecord(null); setQuestion(null); setText(''); setAge(''); setUrgent(''); }}>Start over</button>
            </div>
          </div>

          <aside className="side-column">
            {question ? (
              <div className="question-card">
                <p className="eyebrow">FOLLOW-UP</p><div className="question-mark"><CircleHelp size={20} /></div><h3>{question.question}</h3>
                {question.type === 'B' ? (
                  <>
                    <button className="primary-button full" disabled={loading} onClick={() => respond(question.code, question.code, 'yes')}>Yes <ArrowRight size={16} /></button>
                    <div className="answer-buttons"><button onClick={() => respond(question.code, null, 'no')} disabled={loading}>No</button><button onClick={() => respond(question.code, null, 'unsure')} disabled={loading}>Not sure</button></div>
                  </>
                ) : (
                  <>
                    <select className="text-input" id="follow-value"><option value="">Choose…</option>{question.options?.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select>
                    <button className="primary-button full" onClick={addSelectedValue}>Save response <ArrowRight size={16} /></button>
                  </>
                )}
              </div>
            ) : (
              <div className="info-card"><div className="tool-icon green"><Check size={18} /></div><h3>Follow-up complete</h3><p>Review the two model sections and discuss the result with a clinician.</p></div>
            )}
            <div className="safety-card compact"><ShieldAlert size={18} /><b>Not for emergencies</b><p>Do not use this assessment to rule out urgent or serious illness.</p></div>
          </aside>
        </div>
      )}
    </>
  );
}

function Assistant() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [allowReview, setAllowReview] = useState(false);
  const [loading, setLoading] = useState(false);

  async function sendMessage(event) {
    event.preventDefault();
    if (!text.trim()) return;
    const question = text;
    setMessages((items) => [...items, { role: 'user', text: question }]);
    setText('');
    setLoading(true);
    try {
      const response = await call('/assistant', { method: 'POST', body: JSON.stringify({ message: question, allow_gemini: allowReview }) });
      setMessages((items) => [...items, { role: 'assistant', text: response.answer, sources: response.sources, provider: response.provider }]);
    } catch (error) {
      setMessages((items) => [...items, { role: 'assistant', text: error.message }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="page-heading"><div><p className="eyebrow">GENERAL HEALTH ASSISTANT</p><h1>Ask a question</h1><p className="subhead">Answers are general educational information.</p></div></div>
      <div className="chat-panel">
        <div className="message-list">{messages.map((message, index) => (
          <div className={'message ' + message.role} key={index}>
            <div className="message-avatar">{message.role === 'user' ? 'W' : <Sparkles size={15} />}</div>
            <div className="message-body"><p>{message.text}</p>{message.provider && <small className="message-provider">Optional AI review</small>}</div>
          </div>
        ))}</div>
        <form className="chat-form" onSubmit={sendMessage}><input value={text} onChange={(event) => setText(event.target.value)} placeholder="Ask a health question…" /><button className="primary-button" disabled={loading || !text.trim()}>{loading ? 'Thinking…' : 'Send'}<ArrowRight size={16} /></button></form>
        <label className="consent optional-consent"><input type="checkbox" checked={allowReview} onChange={(event) => setAllowReview(event.target.checked)} /><span>Use optional AI-assisted answers beyond the built-in references.</span></label>
      </div>
    </>
  );
}

function HistoryPage({ rows, onClear }) {
  const [clearing, setClearing] = useState(false);
  async function clearHistory() {
    if (!window.confirm('Delete all saved assessments?')) return;
    setClearing(true);
    try { await call('/history', { method: 'DELETE' }); await onClear(); } catch (error) { window.alert(error.message); } finally { setClearing(false); }
  }
  return (
    <>
      <div className="page-heading"><div><p className="eyebrow">YOUR RECORDS</p><h1>Assessment history</h1></div><button className="secondary-button" disabled={clearing || !rows.length} onClick={clearHistory}>{clearing ? 'Clearing…' : 'Clear history'}</button></div>
      <div className="history-panel">{rows.length ? rows.map((item) => (
        <div className="history-row" key={item.id}><span className="round-icon green"><Activity size={18} /></span><div><b>Symptom assessment</b><small>{new Date(item.created_at).toLocaleString()}</small></div><ChevronRight size={17} /></div>
      )) : <div className="empty-state"><History size={26} /><h3>No saved assessments</h3></div>}</div>
    </>
  );
}

function About() {
  const [data, setData] = useState(null);
  useEffect(() => { call('/about').then(setData).catch(() => {}); }, []);
  return (
    <>
      <div className="page-heading"><div><p className="eyebrow">ABOUT</p><h1>Careful by design</h1></div></div>
      <div className="about-grid">
        <div className="about-main">
          <div className="safety-card"><div className="safety-icon"><ShieldAlert size={19} /></div><p className="card-kicker">MEDICAL DISCLAIMER</p><h2>Educational support, not medical care.</h2><p>{data?.disclaimer || 'This application does not diagnose or replace a licensed clinician.'}</p></div>
          <div className="info-card"><h3>How the two models work</h3><p>DDXPlus ranks synthetic 49-condition evidence. The Indian binary model ranks 30 pattern-based conditions. Neither is clinically validated or locally calibrated.</p></div>
          <div className="info-card"><h3>Your information</h3><p>Assessments are saved only after consent and can be cleared from History. Do not enter identifiable health information in a public deployment.</p></div>
        </div>
      </div>
    </>
  );
}

export default App;
