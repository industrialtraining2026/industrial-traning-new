function InputBar(input, send, onSend) {
  function handleKeyPress(e) {
    if (e.key === 'Enter') {
      handleSend();
    }
  }

  function handleSend() {
    const text = input.value.trim();
    if (text) {
      onSend(text);
      input.value = '';
    }
  }

  input.addEventListener('keypress', handleKeyPress);
  send.addEventListener('click', handleSend);
}



