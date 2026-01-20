window.MessageList = function(container) {
  const el = container;

  const api = {
    push(role, text, id = null) {
      const item = ChatBubble({ role, text, id });
      if (id) {
        item.setAttribute('data-id', id);
      }
      el.appendChild(item);
      el.scrollTop = el.scrollHeight;
    },
    
    removeTyping(typingId) {
      const typingElement = el.querySelector(`[data-id="${typingId}"]`);
      if (typingElement) {
        typingElement.remove();
      }
    }
  }

  return api;
}






