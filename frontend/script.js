// API endpoint URL - automatically detect based on current host
const getApiUrl = () => {
    // If running in browser, use same host with port 8000
    const host = window.location.hostname;
    const protocol = window.location.protocol;
    // Use port 8000 for backend
    return `${protocol}//${host}:8000/evaluate`;
};
const API_URL = getApiUrl();

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
const runButton = document.getElementById('runButton');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressPercentage = document.getElementById('progressPercentage');
const progressStatus = document.getElementById('progressStatus');
const resultsSection = document.getElementById('resultsSection');
const resultsTable = document.getElementById('resultsTable');
const downloadButton = document.getElementById('downloadButton');
let currentResults = null;

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
    
    // Download button
    downloadButton.addEventListener('click', downloadResults);
    
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
    
    // For query, only allow one file (replace existing)
    if (type === 'query') {
        // Clear existing files
        fileArray.length = 0;
        fileList.innerHTML = '';
        // Add only the first file
        if (files.length > 0) {
            const file = files[0];
            fileArray.push(file);
            displayFile(file, fileList, fileArray);
        }
    } else {
        // For database, allow multiple files
        files.forEach(file => {
            // Check if file already exists
            if (!fileArray.find(f => f.name === file.name && f.size === file.size)) {
                fileArray.push(file);
                displayFile(file, fileList, fileArray);
            }
        });
    }
    
    updateRunButtonState();
}

// Display file in list
function displayFile(file, fileList, fileArray) {
    // Create header if it doesn't exist
    let header = fileList.querySelector('.file-list-header');
    if (!header) {
        header = document.createElement('div');
        header.className = 'file-list-header';
        
        const countSpan = document.createElement('span');
        countSpan.className = 'file-list-count';
        countSpan.id = fileList.id + '-count';
        
        const clearButton = document.createElement('button');
        clearButton.className = 'file-list-clear';
        clearButton.textContent = 'Clear All';
        clearButton.addEventListener('click', () => {
            fileArray.length = 0;
            fileList.innerHTML = '';
            updateRunButtonState();
        });
        
        header.appendChild(countSpan);
        header.appendChild(clearButton);
        fileList.insertBefore(header, fileList.firstChild);
    }
    
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    // Check if file is an image
    const isImage = file.type.startsWith('image/');
    
    if (isImage) {
        const img = document.createElement('img');
        img.className = 'file-item-image';
        img.alt = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
        fileItem.appendChild(img);
    } else {
        const icon = document.createElement('div');
        icon.className = 'file-item-icon';
        icon.innerHTML = `
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
        `;
        fileItem.appendChild(icon);
    }
    
    const fileName = document.createElement('span');
    fileName.className = 'file-name';
    fileName.textContent = file.name;
    fileName.title = file.name; // Show full name on hover
    
    const fileSize = document.createElement('span');
    fileSize.className = 'file-size';
    fileSize.textContent = formatFileSize(file.size);
    
    const removeButton = document.createElement('button');
    removeButton.className = 'file-remove';
    removeButton.textContent = '✕';
    removeButton.title = 'Remove';
    removeButton.addEventListener('click', (e) => {
        e.stopPropagation();
        const index = fileArray.indexOf(file);
        if (index > -1) {
            fileArray.splice(index, 1);
            fileItem.remove();
            updateFileCount(fileList, fileArray);
            updateRunButtonState();
        }
    });
    
    fileItem.appendChild(fileName);
    fileItem.appendChild(fileSize);
    fileItem.appendChild(removeButton);
    fileList.appendChild(fileItem);
    
    // Update file count
    updateFileCount(fileList, fileArray);
}

// Update file count display
function updateFileCount(fileList, fileArray) {
    const countElement = fileList.querySelector('.file-list-count');
    if (countElement) {
        const count = fileArray.length;
        countElement.textContent = `${count} file${count !== 1 ? 's' : ''} uploaded`;
    }
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
        alert('Please upload both query image and database files');
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
        formData.append('use_lightglue_only', 'true');  // Always use LightGlue only
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
        let response;
        try {
            response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });
        } catch (fetchError) {
            clearInterval(progressInterval);
            throw new Error(`Failed to connect to backend API at ${API_URL}. Make sure the backend is running on port 8000. Error: ${fetchError.message}`);
        }
        
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
    currentResults = result;
    document.getElementById('resultDatabase').textContent = result.num_database || '-';
    
    // Display results table if results are available
    if (result.results && result.results.length > 0) {
        displayResultsTable(result.results, result.output_dir);
    }
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display results as simple image grid
function displayResultsTable(results, outputDir) {
    // Clear previous results
    resultsTable.innerHTML = '';
    
    // Create a container for visualizations
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '40px';
    container.style.width = '100%';
    container.style.alignItems = 'stretch';
    
    results.forEach((result, idx) => {
        const vizWrapper = document.createElement('div');
        vizWrapper.style.width = '100%';
        vizWrapper.style.marginBottom = '20px';
        vizWrapper.style.display = 'flex';
        vizWrapper.style.justifyContent = 'center';
        
        const vizImg = document.createElement('img');
        vizImg.src = getImageUrl(result.visualization_image, outputDir);
        vizImg.className = 'result-image';
        vizImg.alt = `Query ${result.query_index} visualization`;
        vizImg.title = `Query ${result.query_index} - Click to view full size`;
        vizImg.style.cursor = 'pointer';
        vizImg.style.width = '100%';
        vizImg.style.maxWidth = '100%';
        vizImg.style.height = 'auto';
        vizImg.style.borderRadius = '8px';
        vizImg.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
        vizImg.onclick = () => window.open(vizImg.src, '_blank');
        
        vizWrapper.appendChild(vizImg);
        container.appendChild(vizWrapper);
    });
    
    resultsTable.appendChild(container);
}

// Get image URL from backend
function getImageUrl(imagePath, outputDir) {
    // Extract relative path from output_dir
    let relativePath = imagePath.replace(outputDir, '');
    if (relativePath.startsWith('/')) {
        relativePath = relativePath.substring(1);
    }
    const apiBase = getApiUrl().replace('/evaluate', '');
    return `${apiBase}/outputs/${relativePath}`;
}

// Download results as zip
async function downloadResults() {
    if (!currentResults) {
        alert('No results available to download');
        return;
    }
    
    try {
        downloadButton.disabled = true;
        downloadButton.textContent = '⏳ Preparing download...';
        
        const apiBase = getApiUrl().replace('/evaluate', '');
        const logDir = currentResults.output_dir.split('/').pop();
        const response = await fetch(`${apiBase}/results/${logDir}/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentResults)
        });
        
        if (!response.ok) {
            throw new Error('Failed to download results');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `results_${logDir}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        downloadButton.textContent = '📥 Download Results (ZIP)';
    } catch (error) {
        console.error('Download error:', error);
        alert('Error downloading results: ' + error.message);
        downloadButton.textContent = '📥 Download Results (ZIP)';
    } finally {
        downloadButton.disabled = false;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

