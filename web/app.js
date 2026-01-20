(function(){
  const messages = MessageList(document.getElementById('messages'));
  const input = document.getElementById('user-input');
  const send = document.getElementById('send-btn');
  const suggestionsContainer = document.getElementById('suggestions');
  InputBar(input, send, onSend);

  const API_BASE = 'http://127.0.0.1:8000';
  const HEALTH_CHECK_INTERVAL = 10000; // 10 seconds

  // Static English welcome
  messages.push('bot', "Hi! I'm your Industrial Training assistant. Pick a topic or ask anything.");

  // FAQ categories and pools
  const categories = [
    {
      id: 'overview',
      label: 'Internship Overview 📅',
      keywords: ['internship', 'industrial training', 'overview'],
      pool: [
        'What are the start and end dates of the ITP?',
        'What is the total duration of the ITP in weeks?',
        'What is the minimum credit hour requirement to join the ITP?',
        'Who must a student inform if they relocate to a different branch during the internship?',
        'What student behaviors in the first week are considered poor?',
        'What is the grading standard for the internship?'
      ]
    },
    {
      id: 'formA',
      label: 'Form A 📄',
      keywords: ['form a', 'forma', 'form-a'],
      pool: [
        'What is the submission deadline and time for Form A?',
        'What are the three required documents for Form A submission?',
        'What key information must be shown on the insurance document?',
        'How long does it take to prepare the ITP letter after submitting Form A?',
        'Besides the CV and academic transcript, what must students attach when applying to companies?',
        'What is the minimum number of companies students are advised to apply to?'
      ]
    },
    {
      id: 'formB',
      label: 'Form B 📋',
      keywords: ['form b', 'formb', 'form-b'],
      pool: [
        'What is the submission deadline for Form B?',
        'What are the four documents required for Form B submission?',
        'What specific internship duration must be stated in the company offer letter?',
        'Whose signature is required on the company offer letter besides the sender\'s?',
        'What is the critical submission constraint for Form B?',
        'Can a student change companies once placement is confirmed?'
      ]
    },
    {
      id: 'assessment',
      label: 'Assessment ✅',
      keywords: ['assessment', 'grading', 'evaluation', 'report', 'presentation'],
      pool: [
        'How many supervisors does a student have?',
        'Who is responsible for signing the Weekly Log and completing the Company Evaluation Form?',
        'Who determines the student\'s PASS/FAIL grade?',
        'What will automatically result in an ITP FAIL grade?',
        'When will the Faculty Visitation typically be held?',
        'How often must the Weekly Log be emailed to the Faculty Supervisor?'
      ]
    },
    {
      id: 'company',
      label: 'Company Search 🏢',
      keywords: ['company', 'host', 'placement', 'internship company', 'employer'],
      pool: [
        'What is the minimum number of permanent staff required for a company?',
        'Can the company supervisor be a student\'s close relative?',
        'Is the payment of an allowance by the company mandatory?',
        'Are non-IT jobs like sales or driver allowed for the ITP?',
        'Who must approve a student if they wish to leave their current placement?',
        'Can international students apply to a company in their home country?'
      ]
    }
  ];

  const seen = {};
  categories.forEach(c => { seen[c.id] = new Set(); });

  let pendingCategory = null;

  function findCategoryById(id) {
    return categories.find(c => c.id === id);
  }

  function detectCategory(text = '') {
    const t = text.toLowerCase();
    for (const cat of categories) {
      if (cat.keywords.some(k => t.includes(k))) return cat.id;
    }
    return null;
  }

  function takeNextQuestions(catId, count = 3, exclude = []) {
    const cat = findCategoryById(catId);
    if (!cat) return [];
    const pool = cat.pool;
    const used = seen[catId] || new Set();

    const filtered = pool.filter(q => !used.has(q) && !exclude.includes(q));
    const result = [];

    for (const q of filtered) {
      if (result.length >= count) break;
      result.push(q);
      used.add(q);
    }

    // If not enough, reset and refill (still avoid the immediate exclude and duplicates in result)
    if (result.length < count) {
      used.clear();
      const refill = pool.filter(q => !exclude.includes(q));
      for (const q of refill) {
        if (result.length >= count) break;
        if (!result.includes(q)) {
          result.push(q);
          used.add(q);
        }
      }
    }

    seen[catId] = used;
    return result;
  }

  function renderTopLevel() {
    if (!suggestionsContainer) return;
    suggestionsContainer.innerHTML = '';

    const title = document.createElement('div');
    title.className = 'suggestions-title';
    title.textContent = 'Suggested questions (pick a topic):';
    suggestionsContainer.appendChild(title);

    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'suggestions-buttons';

    categories.forEach(cat => {
      const button = document.createElement('button');
      button.className = 'suggestion-btn';
      button.textContent = cat.label;
      button.addEventListener('click', () => {
        pendingCategory = cat.id;
        renderCategory(cat.id);
      });
      buttonsContainer.appendChild(button);
    });

    suggestionsContainer.appendChild(buttonsContainer);
    suggestionsContainer.style.display = 'block';
  }

  function renderCategory(catId, excludeQuestion = '') {
    if (!suggestionsContainer) return;
    const cat = findCategoryById(catId);
    if (!cat) {
      renderTopLevel();
      return;
    }

    const questions = takeNextQuestions(catId, 3, excludeQuestion ? [excludeQuestion] : []);

    suggestionsContainer.innerHTML = '';
    const header = document.createElement('div');
    header.className = 'suggestions-header';

    const backBtn = document.createElement('button');
    backBtn.className = 'suggestion-back';
    backBtn.textContent = '← Back';
    backBtn.addEventListener('click', () => {
      pendingCategory = null;
      renderTopLevel();
    });

    const title = document.createElement('div');
    title.className = 'suggestions-title';
    title.textContent = `${cat.label} questions:`;

    header.appendChild(backBtn);
    header.appendChild(title);
    suggestionsContainer.appendChild(header);

    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'suggestions-buttons';

    questions.forEach(question => {
      const button = document.createElement('button');
      button.className = 'suggestion-btn';
      button.textContent = question;
      button.addEventListener('click', () => {
        pendingCategory = catId;
        onSend(question, catId);
      });
      buttonsContainer.appendChild(button);
    });

    suggestionsContainer.appendChild(buttonsContainer);
    suggestionsContainer.style.display = 'block';
  }

  function showSuggestionsAfter(text, fallbackCat = null) {
    const cat = pendingCategory || detectCategory(text) || fallbackCat;
    if (cat) {
      renderCategory(cat, text);
    } else {
      renderTopLevel();
    }
  }

  function hideSuggestions() {
    if (suggestionsContainer) {
      suggestionsContainer.style.display = 'none';
    }
  }

  // Initial suggestions
  setTimeout(() => {
    renderTopLevel();
  }, 300);

  // Idle message
  let idleTimer = null;
  const idleMs = 5 * 60 * 1000; // 5 minutes
  function resetIdle() {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      messages.push('bot', "Thanks for chatting! Ask again anytime.");
    }, idleMs);
  }
  resetIdle();

  // Server connection status
  let serverOnline = false;
  let healthCheckInterval = null;

  // Check server health
  async function checkServerHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      
      if (res.ok) {
        if (!serverOnline) {
          // Server just came back online
          serverOnline = true;
          messages.push('bot', "✓ Server is back online. You can ask now.");
        }
        return true;
      }
    } catch (e) {
      // Server is offline
      if (serverOnline) {
        serverOnline = false;
        messages.push('bot', "⚠ Server connection lost. Please ensure the backend is running on http://127.0.0.1:8000");
      }
      return false;
    }
    return false;
  }

  // Start periodic health checks
  function startHealthChecks() {
    // Initial check
    checkServerHealth();
    
    // Periodic checks
    healthCheckInterval = setInterval(checkServerHealth, HEALTH_CHECK_INTERVAL);
  }

  // Stop health checks
  function stopHealthChecks() {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval);
      healthCheckInterval = null;
    }
  }

  // Start health checks when page loads
  startHealthChecks();

  async function onSend(text, forcedCategory = null){
    resetIdle();
    hideSuggestions(); // Hide suggestions when user sends a message
    pendingCategory = forcedCategory || detectCategory(text);
    messages.push('user', text);
    
    // Check server health before sending
    const isHealthy = await checkServerHealth();
    if (!isHealthy) {
      messages.push('bot', "⚠ Cannot connect to the server. Please make sure the backend is running.");
      return;
    }
    
    // Show typing indicator
    const typingId = 'typing-' + Date.now();
    messages.push('bot', '...', typingId);
    
    // Retry mechanism
    let retries = 3;
    let lastError = null;
    
    while (retries > 0) {
      try {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
          signal: AbortSignal.timeout(10000) // 10 second timeout
        });
        
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        
        const data = await res.json();
        
        // Remove typing indicator and add real response
        messages.removeTyping(typingId);
        messages.push('bot', data.reply);
        serverOnline = true; // Mark as online on success
        
        // Show suggestions again after bot responds
        setTimeout(() => {
          showSuggestionsAfter(text, forcedCategory);
        }, 400);
        
        return; // Success, exit retry loop
        
      } catch (e) {
        lastError = e;
        retries--;
        
        if (retries > 0) {
          console.log(`Retry ${3 - retries}/3: ${e.message}`);
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
        }
      }
    }
    
    // All retries failed
    console.error('Chat error after retries:', lastError);
    messages.removeTyping(typingId);
    serverOnline = false;
    
    const errorMsg = `⚠ Connection error: ${lastError.message}. Please ensure the backend is running.`;
    messages.push('bot', errorMsg);
    
    // Show suggestions again after error
    setTimeout(() => {
      showSuggestionsAfter(text, forcedCategory);
    }, 400);
  }
})();
