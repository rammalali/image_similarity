// API endpoint URL
const API_URL = 'http://localhost:8000/evaluate';

// File storage
let queryFiles = [];
let databaseFiles = [];

// DOM elements
const queryUploadArea = document.getElementById('queryUploadArea');
const databaseUploadArea = document.getElementById('databaseUploadArea');
const queryFilesInput = document.getElementById('queryFiles');
const databaseFilesInput = document.getElementById('databaseFiles');
const queryFileList = document.getElementById('queryFileList');
const databaseFileList = document.getElementById('databaseFileList');
const useLightGlueCheckbox = document.getElementById('useLightGlue');
const runButton = document.getElementById('runButton');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressPercentage = document.getElementById('progressPercentage');
const progressStatus = document.getElementById('progressStatus');
const resultsSection = document.getElementById('resultsSection');

// Initialize
function init() {
    // Query upload handlers
    queryUploadArea.addEventListener('click', () => queryFilesInput.click());
    queryFilesInput.addEventListener('change', (e) => handleFileSelect(e, 'query'));
    setupDragAndDrop(queryUploadArea, 'query');
    
    // Database upload handlers
    databaseUploadArea.addEventListener('click', () => databaseFilesInput.click());
    databaseFilesInput.addEventListener('change', (e) => handleFileSelect(e, 'database'));
    setupDragAndDrop(databaseUploadArea, 'database');
    
    // Run button
    runButton.addEventListener('click', runEvaluation);
    
    // Update button state
    updateRunButtonState();
}

// Setup drag and drop
function setupDragAndDrop(area, type) {
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('drag-over');
    });
    
    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });
    
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files);
        addFiles(files, type);
    });
}

// Handle file selection
function handleFileSelect(event, type) {
    const files = Array.from(event.target.files);
    addFiles(files, type);
}

// Add files to list
function addFiles(files, type) {
    const fileArray = type === 'query' ? queryFiles : databaseFiles;
    const fileList = type === 'query' ? queryFileList : databaseFileList;
    
    files.forEach(file => {
        // Check if file already exists
        if (!fileArray.find(f => f.name === file.name && f.size === file.size)) {
            fileArray.push(file);
            displayFile(file, fileList, fileArray);
        }
    });
    
    updateRunButtonState();
}

// Display file in list
function displayFile(file, fileList, fileArray) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const fileName = document.createElement('span');
    fileName.className = 'file-name';
    fileName.textContent = file.name;
    
    const fileSize = document.createElement('span');
    fileSize.className = 'file-size';
    fileSize.textContent = formatFileSize(file.size);
    
    const removeButton = document.createElement('button');
    removeButton.className = 'file-remove';
    removeButton.textContent = 'Remove';
    removeButton.addEventListener('click', () => {
        const index = fileArray.indexOf(file);
        if (index > -1) {
            fileArray.splice(index, 1);
            fileItem.remove();
            updateRunButtonState();
        }
    });
    
    fileItem.appendChild(fileName);
    fileItem.appendChild(fileSize);
    fileItem.appendChild(removeButton);
    fileList.appendChild(fileItem);
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Update run button state
function updateRunButtonState() {
    const hasQueryFiles = queryFiles.length > 0;
    const hasDatabaseFiles = databaseFiles.length > 0;
    runButton.disabled = !(hasQueryFiles && hasDatabaseFiles);
}

// Run evaluation
async function runEvaluation() {
    if (queryFiles.length === 0 || databaseFiles.length === 0) {
        alert('Please upload both query and database files');
        return;
    }
    
    // Disable button and show progress
    runButton.disabled = true;
    runButton.querySelector('.button-text').style.display = 'none';
    runButton.querySelector('.button-loader').style.display = 'inline-block';
    
    // Hide results, show progress
    resultsSection.style.display = 'none';
    progressSection.style.display = 'block';
    updateProgress(0, 'Preparing files...');
    
    try {
        // Prepare form data
        const formData = new FormData();
        
        // Add query files
        queryFiles.forEach(file => {
            formData.append('query_files', file);
        });
        
        // Add database files
        databaseFiles.forEach(file => {
            formData.append('database_files', file);
        });
        
        // Add form parameters
        formData.append('use_lightglue_only', useLightGlueCheckbox.checked);
        formData.append('lightglue_match_threshold', '10');
        formData.append('image_size', '512,512');
        formData.append('recall_values', '1,5,10,20');
        formData.append('num_workers', '4');
        formData.append('batch_size', '4');
        formData.append('device', 'cuda');
        formData.append('log_dir', 'default');
        
        // Simulate progress updates (since we can't track real progress from API)
        const progressInterval = simulateProgress();
        
        // Make API call
        updateProgress(10, 'Uploading files...');
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        updateProgress(90, 'Processing results...');
        const result = await response.json();
        
        updateProgress(100, 'Complete!');
        
        // Show results
        setTimeout(() => {
            showResults(result);
            progressSection.style.display = 'none';
        }, 500);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
        progressSection.style.display = 'none';
    } finally {
        // Re-enable button
        runButton.disabled = false;
        runButton.querySelector('.button-text').style.display = 'inline';
        runButton.querySelector('.button-loader').style.display = 'none';
        updateRunButtonState();
    }
}

// Simulate progress (since we can't track real progress)
function simulateProgress() {
    let currentProgress = 10;
    return setInterval(() => {
        if (currentProgress < 85) {
            currentProgress += Math.random() * 5;
            const statuses = [
                'Extracting descriptors...',
                'Building index...',
                'Finding matches...',
                'Processing images...',
                'Generating visualizations...'
            ];
            const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];
            updateProgress(Math.min(currentProgress, 85), randomStatus);
        }
    }, 2000);
}

// Update progress bar
function updateProgress(percentage, status) {
    progressBar.style.width = percentage + '%';
    progressPercentage.textContent = Math.round(percentage) + '%';
    progressStatus.textContent = status;
}

// Show results
function showResults(result) {
    document.getElementById('resultQueries').textContent = result.num_queries || '-';
    document.getElementById('resultDatabase').textContent = result.num_database || '-';
    document.getElementById('resultOutput').textContent = result.output_dir || '-';
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

