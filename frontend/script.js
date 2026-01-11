// API endpoint URL - automatically detect based on current host
const getApiUrl = () => {
    // If running in browser, use same host with port 8040
    const host = window.location.hostname;
    const protocol = window.location.protocol;
    // Use port 8040 for backend
    return `${protocol}//${host}:8040/evaluate`;
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
    // Query upload handlers - only trigger on placeholder click, not on file list
    queryUploadArea.addEventListener('click', (e) => {
        // Only open file dialog if clicking on the placeholder area, not on file list or buttons
        if (e.target.closest('.upload-placeholder') || 
            (e.target === queryUploadArea && queryFiles.length === 0)) {
            queryFilesInput.click();
        }
    });
    queryFilesInput.addEventListener('change', (e) => handleFileSelect(e, 'query'));
    setupDragAndDrop(queryUploadArea, 'query');
    
    // Database upload handlers - only trigger on placeholder click, not on file list
    databaseUploadArea.addEventListener('click', (e) => {
        // Only open file dialog if clicking on the placeholder area, not on file list or buttons
        if (e.target.closest('.upload-placeholder') || 
            (e.target === databaseUploadArea && databaseFiles.length === 0)) {
            databaseFilesInput.click();
        }
    });
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
    updateFileListHeader(fileList, fileArray);
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
    removeButton.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent event from bubbling to upload area
        e.preventDefault();
        const index = fileArray.indexOf(file);
        if (index > -1) {
            fileArray.splice(index, 1);
            fileItem.remove();
            updateRunButtonState();
            updateFileListHeader(fileList, fileArray);
        }
    });
    
    // Prevent file item clicks from triggering upload dialog
    fileItem.addEventListener('click', (e) => {
        e.stopPropagation();
    });
    
    fileItem.appendChild(fileName);
    fileItem.appendChild(fileSize);
    fileItem.appendChild(removeButton);
    fileList.appendChild(fileItem);
}

// Update file list header with count and clear all button
function updateFileListHeader(fileList, fileArray) {
    // Remove existing header if any
    const existingHeader = fileList.querySelector('.file-list-header');
    if (existingHeader) {
        existingHeader.remove();
    }
    
    // Only show header if there are files
    if (fileArray.length > 0) {
        const header = document.createElement('div');
        header.className = 'file-list-header';
        
        const count = document.createElement('span');
        count.className = 'file-list-count';
        count.textContent = `${fileArray.length} file${fileArray.length > 1 ? 's' : ''}`;
        
        const clearAllButton = document.createElement('button');
        clearAllButton.className = 'file-list-clear';
        clearAllButton.textContent = 'Clear All';
        clearAllButton.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent event from bubbling to upload area
            e.preventDefault();
            fileArray.length = 0;
            fileList.innerHTML = '';
            updateRunButtonState();
        });
        
        header.appendChild(count);
        header.appendChild(clearAllButton);
        fileList.insertBefore(header, fileList.firstChild);
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
        
        // Calculate estimated time: 0.75 seconds per database image
        const numImages = databaseFiles.length;
        const estimatedSeconds = numImages * 0.75;
        const estimatedMs = estimatedSeconds * 1000;
        
        // Start time-based progress bar
        const progressInterval = startTimeBasedProgress(estimatedMs);
        
        // Make API call
        updateProgress(5, 'Uploading files...');
        let response;
        try {
            response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });
        } catch (fetchError) {
            clearInterval(progressInterval);
            throw new Error(`Failed to connect to backend API at ${API_URL}. Make sure the backend is running on port 8040. Error: ${fetchError.message}`);
        }
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        updateProgress(100, 'Complete!');
        const result = await response.json();
        
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

function startTimeBasedProgress(totalMs) {
    const startTime = Date.now();
    const updateInterval = 100; // Update every 100ms for smooth animation
    const statuses = [
        'Extracting features...',
        'Matching images...',
        'Processing matches...',
        'Generating visualizations...'
    ];
    let statusIndex = 0;
    
    return setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(95, (elapsed / totalMs) * 100); // Cap at 95% until complete
        
        // Rotate through status messages
        if (Math.floor(elapsed / 2000) !== Math.floor((elapsed - updateInterval) / 2000)) {
            statusIndex = (statusIndex + 1) % statuses.length;
        }
        
        updateProgress(progress, statuses[statusIndex]);
    }, updateInterval);
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
        const imageUrl = getImageUrl(result.visualization_image, outputDir);
        
        // Clear src first to force browser to reload image
        vizImg.src = '';
        
        // Set image properties
        vizImg.className = 'result-image';
        vizImg.alt = `Query ${result.query_index} visualization`;
        vizImg.title = `Query ${result.query_index} - Click to view full size`;
        vizImg.style.cursor = 'pointer';
        vizImg.style.width = '100%';
        vizImg.style.maxWidth = '100%';
        vizImg.style.height = 'auto';
        vizImg.style.borderRadius = '8px';
        vizImg.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
        
        // Set src after a tiny delay to ensure browser treats it as new image
        setTimeout(() => {
            vizImg.src = imageUrl;
        }, 10);
        
        // Remove cache-busting parameter when opening in new tab (cleaner URL)
        vizImg.onclick = () => {
            const cleanUrl = imageUrl.split('?')[0];
            window.open(cleanUrl, '_blank');
        };
        
        vizWrapper.appendChild(vizImg);
        container.appendChild(vizWrapper);
    });
    
    resultsTable.appendChild(container);
}

// Get image URL from backend with cache-busting
// imagePath format: "default/preds/000.jpg" or similar
// outputDir format: "outputs/default" or similar
function getImageUrl(imagePath, outputDir) {
    const apiBase = getApiUrl().replace('/evaluate', '');
    
    // Extract log_dir and filename from imagePath
    // imagePath is like "default/preds/000.jpg"
    // We need log_dir="default" and filename="preds/000.jpg"
    const pathParts = imagePath.split('/');
    if (pathParts.length >= 2) {
        const logDir = pathParts[0]; // e.g., "default"
        const filename = pathParts.slice(1).join('/'); // e.g., "preds/000.jpg"
        // Use the /results endpoint which has no-cache headers
        // Add cache-busting timestamp to force browser to reload image
        const timestamp = new Date().getTime();
        return `${apiBase}/results/${logDir}/image/${filename}?t=${timestamp}`;
    }
    
    // Fallback to old method if path format is unexpected
    let relativePath = imagePath.replace(outputDir, '');
    if (relativePath.startsWith('/')) {
        relativePath = relativePath.substring(1);
    }
    const timestamp = new Date().getTime();
    return `${apiBase}/outputs/${relativePath}?t=${timestamp}`;
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

