window.ChatBubble = function({ role, text, id }) {
  const row = document.createElement('div');
  row.className = `row ${role}`;

  if (role === 'bot') {
    const avatarWrap = document.createElement('div');
    avatarWrap.className = 'avatar';
    const img = document.createElement('img');
    img.src = 'assets/bot.png';
    img.alt = 'bot';
    img.className = 'bot-avatar';
    avatarWrap.appendChild(img);
    row.appendChild(avatarWrap);
  }

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  
  // Check if this is a typing indicator
  if (text === '...' && role === 'bot') {
    bubble.className += ' typing-indicator';
    bubble.innerHTML = '<span></span><span></span><span></span>';
  } else {
    bubble.textContent = text;
  }
  
  row.appendChild(bubble);

  return row;
}






