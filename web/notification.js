// Use same-origin in production (Render). When opened via file:// fallback to local dev server.
const API_BASE = (window.location.protocol === 'file:') ? 'http://127.0.0.1:8000' : '';

// Check if user is logged in and is a teacher
if (!localStorage.getItem('user_id')) {
    window.location.href = 'login.html';
}

const userType = localStorage.getItem('user_type');
if (userType !== 'teacher') {
    alert('Access denied. Teacher access required.');
    window.location.href = 'home.html';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupEmailUpload();
    loadNotificationStatus();
    loadNotificationHistory();
    loadNotificationPDFList();
    loadEmailFileList();
});

function setupEmailUpload() {
    const uploadArea = document.getElementById('email-upload-area');
    const fileInput = document.getElementById('email-input');
    const fileNameDisplay = document.getElementById('email-file-name');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleEmailFile(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleEmailFile(e.target.files[0]);
        }
    });
}

async function handleEmailFile(file) {
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt')) {
        showMessage('Please select a CSV or TXT file', 'error');
        return;
    }
    
    const fileNameDisplay = document.getElementById('email-file-name');
    fileNameDisplay.textContent = `Selected: ${file.name}`;
    
    await uploadEmailFile(file);
}

async function uploadEmailFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showMessage(`Uploading ${file.name}...`, 'success');
        
        const response = await fetch(`${API_BASE}/api/teacher/upload-emails`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(
                `Successfully uploaded ${result.valid_emails} valid email(s). ${result.invalid_emails} invalid email(s) found.`,
                'success'
            );
            document.getElementById('email-file-name').textContent = '';
            document.getElementById('email-input').value = '';
            loadNotificationStatus();
            // Reload email file list after upload
            setTimeout(() => {
                loadEmailFileList();
            }, 500);
            
            // Show invalid emails if any
            if (result.invalid_emails > 0 && result.invalid_email_list && result.invalid_email_list.length > 0) {
                console.warn('Invalid emails found:', result.invalid_email_list);
                showMessage(
                    `Uploaded ${result.valid_emails} valid email(s). ${result.invalid_emails} invalid email(s) found. Invalid emails: ${result.invalid_email_list.join(', ')}`,
                    result.invalid_emails === result.total_parsed ? 'error' : 'success'
                );
            }
        } else {
            throw new Error(result.error || 'Upload failed');
        }
        
    } catch (error) {
        showMessage(`Upload error: ${error.message}`, 'error');
        console.error('Upload error:', error);
    }
}

async function parseDeadlinePDF() {
    try {
        showMessage('Parsing deadline PDF...', 'success');
        
        const response = await fetch(`${API_BASE}/api/teacher/parse-deadline-pdf`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Display parsed information
        const deadlineInfo = document.getElementById('deadline-info');
        deadlineInfo.innerHTML = `
            <div style="background: #f9fafb; padding: 20px; border-radius: 8px;">
                <h3 style="margin-bottom: 15px; color: #111827;">Parsed Deadline Information</h3>
                <p><strong>Deadline:</strong> ${result.deadline || 'N/A'} ${result.deadline_time || ''}</p>
                ${result.location ? `<p><strong>Location:</strong> ${result.location}</p>` : ''}
                ${result.submission_items && result.submission_items.length > 0 ? `
                    <p><strong>Submission Items:</strong></p>
                    <ul>
                        ${result.submission_items.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                ` : ''}
                ${result.submission_method ? `<p><strong>Submission Method:</strong> ${result.submission_method}</p>` : ''}
            </div>
        `;
        
        showMessage('Deadline PDF parsed successfully', 'success');
        loadNotificationStatus();
        
    } catch (error) {
        showMessage(`Parse error: ${error.message}`, 'error');
        console.error('Parse error:', error);
    }
}

async function loadNotificationStatus() {
    const statusBox = document.getElementById('notification-status');
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/notification-status`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const status = await response.json();
        
        if (!status.has_deadline) {
            statusBox.innerHTML = `
                <p style="color: #9ca3af;">${status.message || 'No deadline information configured'}</p>
                <p style="color: #6b7280; margin-top: 10px; font-size: 14px;">
                    Please upload a notification PDF in Teacher Dashboard and parse it.
                </p>
            `;
            return;
        }
        
        statusBox.innerHTML = `
            <div class="status-item">
                <strong>Deadline:</strong> ${status.deadline_formatted || status.deadline}
            </div>
            <div class="status-item">
                <strong>1-Week Reminder:</strong> 
                ${status.one_week_reminder.sent ? 
                    '<span style="color: #065f46;">✓ Sent</span>' : 
                    `<span style="color: #92400e;">Pending (${status.one_week_reminder.days_until} days)</span>`
                }
            </div>
            <div class="status-item">
                <strong>3-Day Reminder:</strong> 
                ${status.three_days_reminder.sent ? 
                    '<span style="color: #065f46;">✓ Sent</span>' : 
                    `<span style="color: #92400e;">Pending (${status.three_days_reminder.days_until} days)</span>`
                }
            </div>
            <div class="status-item">
                <strong>Student Count:</strong> ${status.student_count || 0} students
            </div>
        `;
        
    } catch (error) {
        statusBox.innerHTML = `<p style="color: #991b1b;">Error loading status: ${error.message}</p>`;
        console.error('Load status error:', error);
    }
}

async function sendNotificationManually() {
    if (!confirm('Are you sure you want to send notification emails to all students?')) {
        return;
    }
    
    try {
        showMessage('Sending notifications...', 'success');
        
        const response = await fetch(`${API_BASE}/api/teacher/send-notification?reminder_type=general`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`Notifications sent successfully. ${result.sent || 0} emails sent.`, 'success');
            loadNotificationStatus();
            loadNotificationHistory();
        } else {
            throw new Error(result.error || 'Send failed');
        }
        
    } catch (error) {
        showMessage(`Send error: ${error.message}`, 'error');
        console.error('Send error:', error);
    }
}

async function loadNotificationPDFList() {
    const fileListContainer = document.getElementById('notification-file-list');
    if (!fileListContainer) return;
    
    fileListContainer.innerHTML = '<div class="file-list-empty">Loading...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/list-pdfs?pdf_type=notification`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success && result.files && result.files.length > 0) {
            fileListContainer.innerHTML = result.files.map(pdf => {
                const escapedFilename = pdf.file_name.replace(/'/g, "\\'");
                return `
                    <div class="file-item">
                        <div class="file-item-name" title="${pdf.file_name}">${pdf.file_name}</div>
                        <div class="file-item-actions">
                            <button class="view-btn" onclick="viewNotificationPDF('${escapedFilename}')">View</button>
                            <button class="delete-btn" onclick="deleteNotificationPDF('${escapedFilename}')">Delete</button>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            fileListContainer.innerHTML = '<div class="file-list-empty">No notification PDF files uploaded yet</div>';
        }
        
    } catch (error) {
        fileListContainer.innerHTML = `<div class="file-list-empty">Error: ${error.message}</div>`;
        console.error('Load notification PDF list error:', error);
    }
}

function viewNotificationPDF(filename) {
    const url = `${API_BASE}/api/teacher/view-pdf?filename=${encodeURIComponent(filename)}&pdf_type=notification`;
    window.open(url, '_blank');
}

async function deleteNotificationPDF(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/delete-pdf?filename=${encodeURIComponent(filename)}&pdf_type=notification`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`Successfully deleted ${result.file_name}`, 'success');
            // Reload PDF list after a short delay
            setTimeout(() => {
                loadNotificationPDFList();
            }, 500);
        } else {
            throw new Error(result.error || 'Delete failed');
        }
        
    } catch (error) {
        showMessage(`Delete error: ${error.message}`, 'error');
        console.error('Delete error:', error);
    }
}

async function loadNotificationHistory() {
    const historyBody = document.getElementById('history-body');
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/notification-history?limit=50`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success && result.history && result.history.length > 0) {
            historyBody.innerHTML = result.history.map(entry => `
                <tr>
                    <td>${entry.deadline_date || 'N/A'}</td>
                    <td>${entry.reminder_type || 'general'}</td>
                    <td>${entry.status || 'N/A'}</td>
                    <td>${entry.recipient_count || 0}</td>
                    <td>${formatDate(entry.sent_time)}</td>
                </tr>
            `).join('');
        } else {
            historyBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #9ca3af;">No notification history</td></tr>';
        }
        
    } catch (error) {
        historyBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #991b1b;">Error loading history: ${error.message}</td></tr>`;
        console.error('Load history error:', error);
    }
}

function showMessage(message, type) {
    const messageEl = document.getElementById('message');
    messageEl.textContent = message;
    messageEl.className = `message ${type}`;
    
    setTimeout(() => {
        messageEl.className = 'message';
    }, 5000);
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (e) {
        return dateString;
    }
}

async function loadEmailFileList() {
    const fileListContainer = document.getElementById('email-file-list');
    if (!fileListContainer) return;
    
    fileListContainer.innerHTML = '<div class="file-list-empty">Loading...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/list-email-files`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success && result.files && result.files.length > 0) {
            fileListContainer.innerHTML = result.files.map(file => {
                const escapedFilename = file.file_name.replace(/'/g, "\\'");
                return `
                    <div class="file-item">
                        <div class="file-item-name" title="${file.file_name}">${file.file_name}</div>
                        <div class="file-item-actions">
                            <button class="view-btn" onclick="viewEmailFile('${escapedFilename}')">View</button>
                            <button class="delete-btn" onclick="deleteEmailFile('${escapedFilename}')">Delete</button>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            fileListContainer.innerHTML = '<div class="file-list-empty">No email files uploaded yet</div>';
        }
        
    } catch (error) {
        fileListContainer.innerHTML = `<div class="file-list-empty">Error: ${error.message}</div>`;
        console.error('Load email file list error:', error);
    }
}

function viewEmailFile(filename) {
    const url = `${API_BASE}/api/teacher/view-email-file?filename=${encodeURIComponent(filename)}`;
    window.open(url, '_blank');
}

async function deleteEmailFile(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/delete-email-file?filename=${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`Successfully deleted ${result.file_name}`, 'success');
            setTimeout(() => {
                loadEmailFileList();
            }, 300);
        } else {
            throw new Error(result.error || 'Delete failed');
        }
        
    } catch (error) {
        showMessage(`Delete error: ${error.message}`, 'error');
        console.error('Delete error:', error);
    }
}

function logout() {
    localStorage.removeItem('user_id');
    localStorage.removeItem('user_type');
    window.location.href = 'login.html';
}

