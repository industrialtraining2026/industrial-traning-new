// Use same-origin in production (Render). When opened via file:// fallback to local dev server.
const API_BASE = (window.location.protocol === 'file:') ? 'http://127.0.0.1:8000' : '';

let currentTab = 'chatbot';
let selectedFiles = {
    chatbot: null,
    submission: null,
    notification: null
};

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
    setupFileUploads();
    loadPDFLists();
});

function switchTab(tabName) {
    currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Load PDF list for current tab
    loadPDFList(tabName);
    if (tabName === 'submission') {
        loadStudentSubmissions();
    }
}

function setupFileUploads() {
    const types = ['chatbot', 'submission', 'notification'];
    
    types.forEach(type => {
        const uploadArea = document.getElementById(`${type}-upload-area`);
        const fileInput = document.getElementById(`${type}-file-input`);
        const fileNameDisplay = document.getElementById(`${type}-file-name`);
        
        // Click to browse
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
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
                handleFileSelect(files[0], type);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0], type);
            }
        });
    });
}

function handleFileSelect(file, type) {
    if (!file.name.endsWith('.pdf')) {
        showMessage('Please select a PDF file', 'error');
        return;
    }
    
    selectedFiles[type] = file;
    const fileNameDisplay = document.getElementById(`${type}-file-name`);
    fileNameDisplay.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    
    // Don't auto upload - user needs to click Upload button
}

function handleManualUpload(type) {
    const file = selectedFiles[type];
    if (!file) {
        showMessage('Please select a file first', 'error');
        return;
    }
    
    uploadPDF(file, type);
}

async function uploadPDF(file, type) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('pdf_type', type);
    
    // Get user_id from localStorage
    const userId = localStorage.getItem('user_id');
    if (userId) {
        formData.append('user_id', userId);
    }
    
    try {
        showMessage(`Uploading ${file.name}...`, 'success');
        
        const response = await fetch(`${API_BASE}/api/teacher/upload-pdf`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`Successfully uploaded ${result.file_name}`, 'success');
            // Clear file selection
            selectedFiles[type] = null;
            const fileNameDisplay = document.getElementById(`${type}-file-name`);
            if (fileNameDisplay) {
                fileNameDisplay.textContent = '';
            }
            const fileInput = document.getElementById(`${type}-file-input`);
            if (fileInput) {
                fileInput.value = '';
            }
            // Reload PDF list after a short delay to ensure file is written
            setTimeout(() => {
                loadPDFList(type);
            }, 500);
        } else {
            throw new Error(result.error || 'Upload failed');
        }
        
    } catch (error) {
        showMessage(`Upload error: ${error.message}`, 'error');
        console.error('Upload error:', error);
    }
}

async function loadPDFLists() {
    loadPDFList('chatbot');
    loadPDFList('submission');
    loadPDFList('notification');
    loadStudentSubmissions();
}

async function loadPDFList(type) {
    const listBody = document.getElementById(`${type}-pdf-list`);
    const fileListContainer = document.getElementById(`${type}-file-list`);
    
    if (!listBody) {
        console.error(`Element ${type}-pdf-list not found`);
        return;
    }
    
    listBody.innerHTML = '<tr><td colspan="5" class="empty-state">Loading...</td></tr>';
    if (fileListContainer) {
        fileListContainer.innerHTML = '<div class="file-list-empty">Loading...</div>';
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/list-pdfs?pdf_type=${type}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('List PDFs API error:', errorText);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        console.log(`Load PDF list for ${type}:`, result);
        
        // Update PDF count display
        const countElement = document.getElementById(`${type}-pdf-count`);
        const typeDisplayNames = {
            'chatbot': 'Chatbot PDF Files',
            'submission': 'Verification CV/Resume PDF Files',
            'notification': 'Notification PDF Files'
        };
        if (countElement) {
            countElement.textContent = `${typeDisplayNames[type]} (${result.count || 0})`;
        }
        
        if (result.success && result.files && result.files.length > 0) {
            // Update file list (above table)
            if (fileListContainer) {
                fileListContainer.innerHTML = result.files.map(pdf => {
                    const escapedFilename = pdf.file_name.replace(/'/g, "\\'");
                    return `
                        <div class="file-item">
                            <div class="file-item-name" title="${pdf.file_name}">${pdf.file_name}</div>
                            <div class="file-item-actions">
                                <button class="view-btn" onclick="viewPDF('${escapedFilename}', '${type}')">View</button>
                                <button class="delete-btn" onclick="deletePDF('${escapedFilename}', '${type}')" style="padding: 4px 8px; font-size: 12px;">Delete</button>
                            </div>
                        </div>
                    `;
                }).join('');
            }
            
            // Update status table
            listBody.innerHTML = result.files.map(pdf => {
                const status = pdf.status || pdf.upload_status || 'pending';
                const statusClass = status === 'success' ? 'status-success' : (status === 'failed' ? 'status-failed' : 'status-pending');
                const statusText = status === 'success' ? 'Success' : (status === 'failed' ? 'Failed' : 'Pending');
                
                // Get last action type
                const lastAction = pdf.last_action || 'none';
                let actionText = 'None';
                if (lastAction === 'upload') {
                    actionText = 'Upload';
                } else if (lastAction === 'delete') {
                    actionText = 'Delete';
                } else if (lastAction === 'rebuild') {
                    actionText = 'Rebuild';
                }
                
                // Escape quotes in filename for onclick
                const escapedFilename = pdf.file_name.replace(/'/g, "\\'");
                
                return `
                    <tr>
                        <td>${pdf.file_name}</td>
                        <td>${formatFileSize(pdf.file_size)}</td>
                        <td>${formatDate(pdf.upload_time)}</td>
                        <td><span class="${statusClass}">${statusText}</span></td>
                        <td>
                            <span style="font-weight: 500; color: #374151;">${actionText}</span>
                            <button class="delete-btn" onclick="deletePDF('${escapedFilename}', '${type}')" style="margin-left: 10px;">Delete</button>
                        </td>
                    </tr>
                `;
            }).join('');
        } else {
            if (fileListContainer) {
                fileListContainer.innerHTML = '<div class="file-list-empty">No PDF files uploaded yet</div>';
            }
            listBody.innerHTML = '<tr><td colspan="5" class="empty-state">No PDF files uploaded yet</td></tr>';
        }
        
    } catch (error) {
        if (fileListContainer) {
            fileListContainer.innerHTML = `<div class="file-list-empty">Error: ${error.message}</div>`;
        }
        listBody.innerHTML = `<tr><td colspan="5" class="empty-state">Error loading PDFs: ${error.message}</td></tr>`;
        console.error('Load PDF list error:', error);
    }
}

async function loadStudentSubmissions() {
    const listBody = document.getElementById('student-submission-pdf-list');
    const countEl = document.getElementById('student-submission-pdf-count');

    if (!listBody) {
        console.error('Element student-submission-pdf-list not found');
        return;
    }

    listBody.innerHTML = '<tr><td colspan="5" class="empty-state">Loading...</td></tr>';

    try {
        const response = await fetch(`${API_BASE}/api/teacher/list-student-submissions`);

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const result = await response.json();

        if (countEl) {
            countEl.textContent = `Student Submission CV/Resume PDFs (${result.count || 0})`;
        }

        if (result.success && result.files && result.files.length > 0) {
            listBody.innerHTML = result.files.map(pdf => {
                const escapedFilename = pdf.file_name.replace(/'/g, "\\'");
                return `
                    <tr>
                        <td>${pdf.file_name}</td>
                        <td>${pdf.uploaded_by || '-'}</td>
                        <td>${formatDate(pdf.upload_time)}</td>
                        <td>${formatFileSize(pdf.file_size)}</td>
                        <td>
                            <button class="view-btn" onclick="viewPDF('${escapedFilename}', 'submission')">View</button>
                        </td>
                    </tr>
                `;
            }).join('');
        } else {
            listBody.innerHTML = '<tr><td colspan="5" class="empty-state">No student submissions yet</td></tr>';
        }
    } catch (error) {
        listBody.innerHTML = `<tr><td colspan="5" class="empty-state">Error loading student submissions: ${error.message}</td></tr>`;
        console.error('Load student submissions error:', error);
    }
}

function viewPDF(filename, type) {
    const url = `${API_BASE}/api/teacher/view-pdf?filename=${encodeURIComponent(filename)}&pdf_type=${type}`;
    window.open(url, '_blank');
}

async function deletePDF(filename, type) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/teacher/delete-pdf?filename=${encodeURIComponent(filename)}&pdf_type=${type}`, {
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
                loadPDFList(type);
            }, 300);
        } else {
            throw new Error(result.error || 'Delete failed');
        }
        
    } catch (error) {
        showMessage(`Delete error: ${error.message}`, 'error');
        console.error('Delete error:', error);
    }
}

function showMessage(message, type) {
    const messageEl = document.getElementById('message');
    messageEl.textContent = message;
    messageEl.className = `message ${type}`;
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        messageEl.className = 'message';
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
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

async function rebuildFAISSIndex(pdf_type) {
    const typeDisplayNames = {
        'chatbot': 'Chatbot',
        'submission': 'Verification CV/Resume',
        'notification': 'Notification'
    };
    
    if (!confirm(`Are you sure you want to rebuild the FAISS index for ${typeDisplayNames[pdf_type]} PDFs? This will clear the existing index and reindex all PDFs in this category. This may take a while.`)) {
        return;
    }
    
    try {
        showMessage(`Rebuilding FAISS index for ${typeDisplayNames[pdf_type]} PDFs...`, 'success');
        
        const response = await fetch(`${API_BASE}/api/teacher/rebuild-faiss-index?pdf_type=${pdf_type}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`FAISS index rebuilt successfully. Processed ${result.result?.processed_files || 0} files.`, 'success');
            // Reload PDF list to update status
            loadPDFList(pdf_type);
        } else {
            throw new Error(result.error || 'Rebuild index failed');
        }
        
    } catch (error) {
        showMessage(`Rebuild index error: ${error.message}`, 'error');
        console.error('Rebuild index error:', error);
    }
}

function logout() {
    localStorage.removeItem('user_id');
    localStorage.removeItem('user_type');
    window.location.href = 'login.html';
}






