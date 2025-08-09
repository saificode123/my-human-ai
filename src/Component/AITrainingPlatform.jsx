import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Upload,
  FileText,
  Database,
  Code,
  X,
  CheckCircle,
  AlertCircle,
  Play,
  RotateCcw,
  Brain,
  Settings,
  BarChart3,
  FileUp,
  FileImage,
  Loader2,
  User,
  MessageSquare,
  Send,
  Star,
  Zap,
  Target,
  Activity
} from 'lucide-react';

// Mock API Service - In a real app, this would be an actual API client.
// For this standalone frontend, we'll keep it simple, but imagine it calls your backend.
const apiService = {
  getTrainingStatus: async (userId) => {
    // Simulate API call to backend /api/training/status/{user_id}
    const response = await fetch(`http://localhost:8000/api/training/status/${userId}`);
    if (!response.ok) {
      throw new Error(`Error fetching training status: ${response.statusText}`);
    }
    return response.json();
  },
  getUserStats: async (userId) => { // This now correctly points to /api/stats
    // Simulate API call to backend /api/stats/{user_id}
    const response = await fetch(`http://localhost:8000/api/stats/${userId}`);
    if (!response.ok) {
      throw new Error(`Error fetching user stats: ${response.statusText}`);
    }
    return response.json();
  },
  startTraining: async (userId, modelName) => {
    // Simulate API call to backend /api/training/start
    const response = await fetch(`http://localhost:8000/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, model_name: modelName, include_social_media: true }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to start training: ${errorData.detail || response.statusText}`);
    }
    return response.json();
  },
  sendChatMessage: async (userId, message, useContext, useOpenAIPolish) => {
    // Simulate API call to backend /api/chat
    const response = await fetch(`http://localhost:8000/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, message, use_context: useContext, use_openai_polish: useOpenAIPolish }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to send message: ${errorData.detail || response.statusText}`);
    }
    return response.json();
  },
  createUser: async () => {
    const response = await fetch(`http://localhost:8000/api/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`Failed to create user: ${response.statusText}`);
    }
    return response.json();
  },
  uploadFile: async (userId, file) => { // Simplified, removed trainingConfig
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('files', file); // Use 'files' as the key for consistency with backend
    const response = await fetch(`http://localhost:8000/api/upload`, {
      method: 'POST',
      body: formData, // No Content-Type header needed for FormData, fetch sets it automatically
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to upload file: ${errorData.detail || response.statusText}`);
    }
    return response.json();
  },
};


const AVAILABLE_MODELS = [
  'distilgpt2',
  'gpt2',
  'microsoft/DialoGPT-small',
  'facebook/blenderbot-400M-distill'
];

const TASK_TYPES = [
  'text-generation',
  'classification',
  'sentiment-analysis',
  'question-answering',
  'summarization'
];

// File Upload Component
// Modified to handle multiple file uploads
const FileUploadArea = ({ onFilesSelect, selectedFiles, onRemoveFile, isUploading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    onFilesSelect(files);
  }, [onFilesSelect]);

  const handleFileChange = useCallback((e) => {
    const files = Array.from(e.target.files);
    onFilesSelect(files);
  }, [onFilesSelect]);

  const allowedFileTypes = ['.txt', '.pdf', '.docx'];

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors
        ${isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50'}
        ${uploadError ? 'border-red-500 bg-red-50' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        multiple
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".txt,.pdf,.docx,.zip" // Added .zip for LinkedIn export
      />
      <FileUp className="mx-auto w-12 h-12 text-gray-400 mb-3" />
      <p className="text-gray-600">Drag & drop files here, or <button onClick={() => fileInputRef.current.click()} className="text-blue-600 font-medium hover:underline">browse</button></p>
      <p className="text-sm text-gray-500 mt-1">Supported: .txt, .pdf, .docx, .zip (for LinkedIn)</p>
      {uploadError && (
        <p className="text-red-500 text-sm mt-2 flex items-center justify-center gap-1">
          <AlertCircle className="w-4 h-4" /> {uploadError}
        </p>
      )}

      {selectedFiles.length > 0 && (
        <div className="mt-4 border-t border-gray-200 pt-4">
          <h4 className="font-semibold text-gray-700 mb-2">Selected Files:</h4>
          <ul className="text-left space-y-2">
            {selectedFiles.map((file, index) => (
              <li key={index} className="flex items-center justify-between bg-white p-2 rounded-md shadow-sm border border-gray-100">
                <span className="text-sm text-gray-800 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-500" />
                  {file.name} ({Math.round(file.size / 1024)} KB)
                </span>
                <button
                  onClick={() => onRemoveFile(index)}
                  className="text-red-500 hover:text-red-700 p-1 rounded-full hover:bg-red-100 transition-colors"
                  disabled={isUploading}
                >
                  <X className="w-4 h-4" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {isUploading && (
        <div className="mt-4 flex items-center justify-center text-blue-600">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          <span>Uploading...</span>
        </div>
      )}
    </div>
  );
};


// Debugger Modal Component
const AIDebugger = ({ trainingStatus, selectedFiles, metadata, chatHistory, userId, onClose }) => {
  const [activeTab, setActiveTab] = useState("realtime");
  const [loading, setLoading] = useState(false);
  // Default user ID for demonstration. In a real app, this might come from auth or be created.
  const [currentUserId, setCurrentUserId] = useState(userId || "demo-user-y2qjjyb4a"); 
  const [apiBase, setApiBase] = useState("http://localhost:8000"); // Your backend API base URL
  const [realData, setRealData] = useState({
    trainingStatus: null,
    userData: null, 
    modelExists: false,
    apiHealth: null,
    lastUpdated: null,
    errors: []
  });

  // Fetch real-time data from your APIs
  const fetchRealTimeData = async () => {
    setLoading(true);
    const errors = [];
    
    try {
      // 1. Check API health
      const healthResponse = await fetch(`${apiBase}/api/health`); 
      if (!healthResponse.ok) throw new Error(`Health check failed: ${healthResponse.statusText}`);
      const healthData = await healthResponse.json();
      
      // 2. Get training status
      const trainingResponse = await fetch(`${apiBase}/api/training/status/${currentUserId}`);
      let trainingData = null;
      if (trainingResponse.ok) {
        trainingData = await trainingResponse.json();
      } else if (trainingResponse.status === 404) {
        console.warn(`Training status not found for user ${currentUserId}. Assuming no training started.`);
      } else {
        throw new Error(`Training status fetch failed: ${trainingResponse.statusText}`);
      }

      // 3. Get user data (which includes clone info, personality, etc.)
      // CORRECTED ENDPOINT HERE:
      const userDataResponse = await fetch(`${apiBase}/api/stats/${currentUserId}`); // ✨ FIXED THIS LINE ✨
      
      let userData = null;
      if (userDataResponse.ok) {
        userData = await userDataResponse.json();
      } else if (userDataResponse.status === 404) {
        console.warn(`User data not found for user ${currentUserId}.`);
      } else {
        throw new Error(`User data fetch failed: ${userDataResponse.statusText}`);
      }
      
      // Determine model existence based on backend response (using new data structure from /api/stats)
      // The /api/stats endpoint returns 'total_documents_in_chroma' and 'training_status'
      const modelExists = (userData?.total_documents_in_chroma > 0 && userData?.training_status === 'completed');

      setRealData({
        trainingStatus: trainingData,
        userData: userData, // Now storing data from /api/stats
        modelExists: modelExists,
        apiHealth: healthData,
        lastUpdated: new Date().toLocaleString(),
        errors: []
      });
      
    } catch (error) {
      errors.push(`API Connection Failed: ${error.message}`);
      setRealData(prev => ({ ...prev, errors, lastUpdated: new Date().toLocaleString() }));
    }
    
    setLoading(false);
  };

  // Auto-refresh every 10 seconds
  useEffect(() => {
    fetchRealTimeData();
    const interval = setInterval(fetchRealTimeData, 10000);
    return () => clearInterval(interval);
  }, [currentUserId, apiBase]); // Re-fetch if currentUserId or apiBase changes

  // Helper component for status indicators
  const StatusIndicator = ({ status }) => {
    const getStatusColor = (status) => {
      switch (status) {
        case 'completed': return 'text-green-500';
        case 'training': case 'preprocessing': case 'fetching': case 'creating_embeddings': case 'processing_files': case 'processing_social_media': return 'text-blue-500'; // Added processing stages
        case 'error': case 'failed': return 'text-red-500';
        case 'not_started': case 'not_found': return 'text-yellow-500'; // For 'not_started' or 'not_found'
        default: return 'text-gray-500'; 
      }
    };
    
    const getStatusIcon = (status) => {
      switch (status) {
        case 'completed': return <CheckCircle className="w-4 h-4" />;
        case 'error': case 'failed': return <XCircle className="w-4 h-4" />;
        case 'training': case 'preprocessing': case 'fetching': case 'creating_embeddings': case 'processing_files': case 'processing_social_media': return <Activity className="w-4 h-4 animate-spin" />; // Spinning for active
        default: return <AlertTriangle className="w-4 h-4" />; // For 'unknown' or 'pending'
      }
    };
    
    const displayStatus = status || "unknown"; // Default to 'unknown' if status is null/undefined

    return (
      <div className={`flex items-center gap-2 ${getStatusColor(displayStatus)}`}>
        {getStatusIcon(displayStatus)}
        <span className="capitalize font-medium">{displayStatus.replace(/_/g, ' ')}</span>
      </div>
    );
  };

  // Helper component for displaying metrics
  const MetricCard = ({ title, value, status = "info", description }) => (
    <div className="bg-white rounded-xl p-4 shadow-sm border">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        <StatusIndicator status={status} />
      </div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      {description && <p className="text-xs text-gray-500 mt-1">{description}</p>}
    </div>
  );

  // Helper component for displaying alerts/issues
  const IssueAlert = ({ type, title, description, severity = "warning" }) => {
    const severityColors = {
      error: "bg-red-50 border-red-200 text-red-800",
      warning: "bg-yellow-50 border-yellow-200 text-yellow-800",
      info: "bg-blue-50 border-blue-200 text-blue-800",
      success: "bg-green-50 border-green-200 text-green-800"
    };
    
    return (
      <div className={`p-4 rounded-lg border ${severityColors[severity]} mb-3`}>
        <div className="flex items-start gap-3">
          <div className="font-semibold">{title}</div>
        </div>
        <p className="mt-1 text-sm">{description}</p>
      </div>
    );
  };

  // Analyze real data for issues and generate recommendations
  const analyzeRealData = () => {
    const issues = [];
    
    if (!realData.apiHealth || realData.apiHealth.status !== 'healthy') {
      issues.push({
        type: "connection",
        title: "❌ API Connection Failed",
        description: "Cannot connect to your AI training platform. Check if the server is running on the correct port.",
        severity: "error"
      });
    }
    
    // Check if training status is available and not an error
    if (realData.trainingStatus) {
      if (realData.trainingStatus.status === 'failed' || realData.trainingStatus.status === 'error') {
        issues.push({
          type: "training_failed",
          title: "🚫 Training Failed",
          description: realData.trainingStatus.message || "Training process encountered an error",
          severity: "error"
        });
      } else if (realData.trainingStatus.status === 'not_started' || realData.trainingStatus.progress === 0) {
        // If training is pending or at 0% and no user data (meaning no profile submitted)
        if (!realData.userData || realData.userData.total_documents_in_chroma === 0) {
          issues.push({
            type: "nodata",
            title: "📁 No Training Data Submitted",
            description: "No uploaded files or social profiles processed for this user. Upload data and submit a profile to start training.",
            severity: "warning"
          });
        }
      }
    } else {
        // If trainingStatus is null, it means no training has been initiated or found for the user
        issues.push({
            type: "training_not_found",
            title: "❓ No Training Session Found",
            description: "No active or past training session found for this user ID. You might need to create a user and submit a profile to start training.",
            severity: "info"
        });
    }

    if (realData.userData && realData.userData.total_documents_in_chroma < 50 && realData.trainingStatus?.status === 'completed') { 
      issues.push({
        type: "insufficient",
        title: "⚠️ Insufficient Training Data",
        description: `Only ${realData.userData.total_documents_in_chroma || 0} chunks found. Need 50+ chunks for good results.`,
        severity: "warning"
      });
    }
    
    if (realData.trainingStatus?.status === 'completed' && !realData.modelExists) {
      issues.push({
        type: "model_files_missing",
        title: "🤖 Model Files Missing",
        description: "Training shows complete but model files (ChromaDB collection) are missing. Check backend logs and ChromaDB directory.",
        severity: "error"
      });
    }
    
    return issues;
  };

  const realTimeIssues = analyzeRealData();

  const tabs = [
    { id: "realtime", label: "📊 Real-Time Status", icon: Activity },
    { id: "training", label: "🎯 Training Analysis", icon: Brain },
    { id: "model", label: "🤖 Model Data", icon: Database },
    { id: "fixes", label: "🔧 Live Fixes", icon: Settings },
  ];

  return (
    <div className="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center p-4 z-50 overflow-y-auto font-inter">
      <div className="relative bg-gradient-to-br from-blue-50 to-indigo-100 rounded-2xl shadow-xl p-8 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 transition-colors p-2 rounded-full bg-white shadow-md"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🔍 Real-Time AI Training Debugger
          </h1>
          <p className="text-gray-600">
            Live analysis of your Human Clone AI platform • Last updated: {realData.lastUpdated || 'N/A'}
          </p>
        </div>

        {/* Configuration Panel */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6 border border-gray-200">
          <h3 className="font-semibold mb-3 text-gray-700">🔧 Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">API Base URL</label>
              <input
                type="text"
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                placeholder="http://localhost:8000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">User ID</label>
              <input
                type="text"
                value={currentUserId}
                onChange={(e) => setCurrentUserId(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                placeholder="demo-user-y2qjjyb4a"
              />
            </div>
          </div>
          <button
            onClick={fetchRealTimeData}
            disabled={loading}
            className="mt-4 px-5 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2 transition-colors shadow-md"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Refreshing...' : 'Refresh Data'}
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto border-b border-gray-200">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-5 py-2 rounded-t-lg font-medium transition-colors whitespace-nowrap text-base ${
                activeTab === tab.id
                  ? "bg-indigo-600 text-white shadow-lg"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Contents */}
        {activeTab === "realtime" && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Training Status"
                value={realData.trainingStatus?.status || "Not Found"}
                status={realData.trainingStatus?.status}
                description={realData.trainingStatus?.message || "No status message available."}
              />
              <MetricCard
                title="Progress"
                value={`${realData.trainingStatus?.progress || 0}%`}
                status={realData.trainingStatus?.progress === 100 ? "completed" : "training"}
                description="Current progress of the training process."
              />
              <MetricCard
                title="Model Exists (Chroma)"
                value={realData.modelExists ? "Yes" : "No"}
                status={realData.modelExists ? "completed" : "error"}
                description="Indicates if the trained ChromaDB collection is present and has data."
              />
              <MetricCard
                title="API Health"
                value={realData.apiHealth?.status || "Unknown"}
                status={realData.apiHealth?.status}
                description={realData.apiHealth?.timestamp ? `Last checked: ${new Date(realData.apiHealth.timestamp).toLocaleTimeString()}` : "API health status."}
              />
            </div>

            {/* Live Issues */}
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">🚨 Live Issues Detected ({realTimeIssues.length})</h3>
              {realTimeIssues.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <CheckCircle className="w-12 h-12 mx-auto mb-2 text-green-500" />
                  <p>No critical issues detected!</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {realTimeIssues.map((issue, idx) => (
                    <IssueAlert key={idx} {...issue} />
                  ))}
                </div>
              )}
            </div>

            {/* Connection Errors */}
            {realData.errors.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <h3 className="text-red-800 font-semibold mb-2">🔌 Connection Errors</h3>
                {realData.errors.map((error, idx) => (
                  <p key={idx} className="text-red-700 text-sm">{error}</p>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === "training" && (
          <div className="space-y-6">
            {/* Training Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricCard
                title="Total Chunks Processed"
                value={realData.userData?.total_chunks_processed || 0}
                status={realData.userData?.total_chunks_processed > 50 ? "completed" : "warning"}
                description="Minimum 50 chunks recommended for effective training."
              />
              <MetricCard
                title="Uploaded Files Count"
                value={realData.userData?.uploaded_files_count || 0}
                status={realData.userData?.uploaded_files_count > 0 ? "completed" : "info"}
                description="Number of files uploaded for training."
              />
              <MetricCard
                title="Social Data Files Count"
                value={realData.userData?.social_data_files_count || 0}
                status={realData.userData?.social_data_files_count > 0 ? "completed" : "info"}
                description="Number of social media data files processed."
              />
            </div>

            {/* Training Details */}
            {realData.trainingStatus ? (
              <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">📈 Training Details</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Model Name (Backend):</span>
                    <span className="text-gray-900">{realData.trainingStatus.model_name || "Not specified"}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Embedding Model:</span>
                    <span className="text-gray-900">{realData.trainingStatus.embedding_model || "Not specified"}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Progress:</span>
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${realData.trainingStatus.progress || 0}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Status Message:</span>
                    <span className="text-sm text-gray-600">{realData.trainingStatus.message}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Training Started:</span>
                    <span className="text-sm text-gray-600">{realData.userData?.start_time || "N/A"}</span>
                  </div>
                </div>
              </div>
            ) : (
                <div className="bg-gray-50 rounded-lg p-6 border border-gray-200 text-center py-8 text-gray-500">
                    <Brain className="w-12 h-12 mx-auto mb-2" />
                    <p>No training details available. Submit a profile to start training.</p>
                </div>
            )}
          </div>
        )}

        {activeTab === "model" && (
          <div className="space-y-6">
            {/* Model Status */}
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">🤖 Model Information</h3>
              {realData.userData && realData.userData.total_documents_in_chroma > 0 ? (
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-auto max-h-96">
                  <pre>{JSON.stringify(realData.userData, null, 2)}</pre>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Database className="w-12 h-12 mx-auto mb-2" />
                  <p>No model metadata available</p>
                  <p className="text-sm">Train your model first to see details</p>
                </div>
              )}
            </div>

            {/* Model Files Check */}
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">📁 Model Files Status</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className={`p-3 rounded-lg ${realData.modelExists ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                  <div className="flex items-center gap-2">
                    {realData.modelExists ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                    <span className="font-medium">ChromaDB Collection</span>
                  </div>
                  <p className="text-sm mt-1">
                    {realData.modelExists ? 'Found and populated' : 'Missing or empty'}
                  </p>
                </div>
                
                <div className="p-3 rounded-lg bg-blue-100 text-blue-800">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    <span className="font-medium">Expected Chroma Path</span>
                  </div>
                  <p className="text-sm mt-1 font-mono">
                    ./data/chromadb/user_{currentUserId}/
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "fixes" && (
          <div className="space-y-6">
            {/* Quick Actions */}
            <div className="bg-green-50 border-2 border-green-200 rounded-xl p-6">
              <h3 className="text-green-800 font-bold text-lg mb-4">⚡ Quick Actions</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={fetchRealTimeData} 
                  className="p-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md flex flex-col items-center justify-center"
                >
                  <RefreshCw className="w-5 h-5 mx-auto mb-2" />
                  Refresh All Data
                </button>
                
                <button
                  onClick={async () => {
                    try {
                      // Corrected endpoint to /api/training/start and pass model_name
                      const response = await apiService.startTraining(currentUserId, 'distilgpt2'); 
                      alert("Training initiated successfully! Refresh status to see progress."); 
                      fetchRealTimeData(); 
                    } catch (error) {
                      alert(`Error initiating training: ${error.message}`); 
                      console.error("Error initiating training:", error);
                    }
                  }}
                  className="p-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors shadow-md flex flex-col items-center justify-center"
                >
                  <Brain className="w-5 h-5 mx-auto mb-2" />
                  Start Training (Submit Profile)
                </button>
              </div>
            </div>

            {/* Live Recommendations */}
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">💡 Live Recommendations</h3>
              
              {realTimeIssues.length === 0 ? (
                <div className="text-center py-4 text-gray-500">
                  <p>No specific recommendations at this moment. All looks good!</p>
                </div>
              ) : (
                realTimeIssues.map((issue, idx) => (
                  <div key={idx} className="mb-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
                    <h4 className="font-medium text-gray-900 mb-2">{issue.title}</h4>
                    <p className="text-sm text-gray-600 mb-3">{issue.description}</p>
                    
                    {issue.type === 'insufficient' && (
                      <div className="text-xs bg-blue-50 p-2 rounded text-blue-800">
                        <strong>Fix:</strong> Upload more personal data files or add detailed social media information (aim for 50+ chunks).
                      </div>
                    )}
                    
                    {issue.type === 'connection' && (
                      <div className="text-xs bg-blue-50 p-2 rounded text-blue-800">
                        <strong>Fix:</strong> Ensure your backend server is running. In your terminal, navigate to the backend directory and run: <code className="bg-gray-200 px-1 rounded">uvicorn main:app --reload --host 0.0.0.0 --port 8000</code>
                      </div>
                    )}
                    
                    {issue.type === 'model_files_missing' && (
                      <div className="text-xs bg-blue-50 p-2 rounded text-blue-800">
                        <strong>Fix:</strong> Check the `data/chromadb` directory in your backend for the user's collection. If missing, restart training.
                      </div>
                    )}

                    {issue.type === 'nodata' && (
                      <div className="text-xs bg-blue-50 p-2 rounded text-blue-800">
                        <strong>Fix:</strong> Use the 'Upload Data' section to upload files, then go to the 'Live Fixes' tab and click 'Start Training (Submit Profile)'.
                      </div>
                    )}

                    {issue.type === 'training_not_found' && (
                      <div className="text-xs bg-blue-50 p-2 rounded text-blue-800">
                        <strong>Fix:</strong> Create a user via the backend API (e.g., POST to `/api/users`) and then submit a profile via the 'Start Training' button.
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>

            {/* System Commands */}
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
              <h4 className="text-white font-medium mb-3">🖥️ Debug Commands (Run in your backend terminal)</h4>
              <div className="space-y-2 text-sm font-mono">
                <div># Check ChromaDB directory for the current user's collection:</div>
                <div className="bg-gray-800 p-2 rounded">ls -la ./data/chromadb/user_{currentUserId}/</div>
                <div className="mt-3"># Check the application logs for detailed errors:</div>
                <div className="bg-gray-800 p-2 rounded">tail -f app.log</div>
                <div className="mt-3"># Test API health directly:</div>
                <div className="bg-gray-800 p-2 rounded">curl {apiBase}/api/health</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};


// Main App component
const AITrainingPlatform = () => {
  const [userId, setUserId] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState({});
  const [chatHistory, setChatHistory] = useState([]);
  const [message, setMessage] = useState('');
  const [useOpenAIPolish, setUseOpenAIPolish] = useState(false);
  const [showDebugger, setShowDebugger] = useState(false);
  const [isCreatingUser, setIsCreatingUser] = useState(false);

  // Mock metadata for demonstration. In a real app, this might come from user input.
  const [metadata, setMetadata] = useState({
    modelName: AVAILABLE_MODELS[0],
    taskType: TASK_TYPES[0],
  });

  // Function to create a new user
  const handleCreateUser = async () => {
    setIsCreatingUser(true);
    try {
      const response = await apiService.createUser();
      setUserId(response.user_id);
      alert(`New User ID created: ${response.user_id}`); // Use a proper modal for UX
    } catch (error) {
      alert(`Error creating user: ${error.message}`);
      console.error("Error creating user:", error);
    } finally {
      setIsCreatingUser(false);
    }
  };

  const handleFilesSelect = (files) => {
    setSelectedFiles(prev => [...prev, ...files]);
  };

  const handleRemoveFile = (indexToRemove) => {
    setSelectedFiles(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const handleUploadFiles = async () => {
    if (selectedFiles.length === 0 || !userId) {
      alert('Please select files and ensure a user ID is created first.');
      return;
    }
    setIsUploading(true);
    try {
      // Upload files one by one (or in batches)
      for (const file of selectedFiles) {
        await apiService.uploadFile(userId, file);
      }
      alert('Files uploaded successfully! Now you can start training.');
      setSelectedFiles([]); // Clear selected files after successful upload
    } catch (error) {
      alert(`Failed to upload files: ${error.message}`);
      console.error("Upload error:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const startTraining = async () => {
    if (!userId) {
      alert('Please create a user first.');
      return;
    }
    try {
      const response = await apiService.startTraining(userId, metadata.modelName);
      alert(`Training initiated: ${response.message}`);
      // The debugger will pick up status changes automatically
    } catch (error) {
      alert(`Failed to start training: ${error.message}`);
      console.error("Training initiation error:", error);
    }
  };

  const handleSendMessage = async () => {
    if (!message.trim() || !userId) return;

    const userMessage = { sender: 'user', text: message };
    setChatHistory(prev => [...prev, userMessage]);
    setMessage('');

    try {
      const response = await apiService.sendChatMessage(userId, message, true, useOpenAIPolish);
      const botMessage = { sender: 'bot', text: response.response, context: response.context_used, score: response.similarity_score, sources: response.sources };
      setChatHistory(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { sender: 'bot', text: `Error: ${error.message}` };
      setChatHistory(prev => [...prev, errorMessage]);
      console.error("Chat error:", error);
    }
  };

  // Poll for training status updates (for the main app's display, not the debugger)
  useEffect(() => {
    let interval;
    if (userId) {
      interval = setInterval(async () => {
        try {
          const status = await apiService.getTrainingStatus(userId);
          setTrainingStatus(status);
        } catch (error) {
          console.error("Error fetching training status:", error);
          // Optionally set an error state here
        }
      }, 5000); // Poll every 5 seconds
    }
    return () => clearInterval(interval);
  }, [userId]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 p-8 font-inter text-gray-800">
      <div className="max-w-6xl mx-auto bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 space-y-8 border border-gray-200">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-700 mb-4">
            Human Clone AI Trainer
          </h1>
          <p className="text-lg text-gray-600">Personalize Your AI Experience with Your Data</p>
        </div>

        {/* User Management & Status */}
        <div className="bg-gray-50 rounded-xl p-6 shadow-inner border border-gray-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          {!userId ? (
            <div className="flex items-center gap-3 text-red-500 font-medium">
              <AlertCircle className="w-5 h-5" />
              <span>No User ID. Please create one to start.</span>
              <button
                onClick={handleCreateUser}
                className="ml-4 px-6 py-2 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition-colors flex items-center gap-2"
                disabled={isCreatingUser}
              >
                {isCreatingUser ? <Loader2 className="animate-spin w-5 h-5" /> : <User className="w-5 h-5" />}
                {isCreatingUser ? 'Creating...' : 'Create New User'}
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-4 text-gray-700 font-medium">
              <User className="w-5 h-5 text-green-600" />
              <span>User ID: <span className="font-mono text-sm bg-gray-200 px-2 py-1 rounded">{userId}</span></span>
              <span className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-600" />
                Training Status: <span className={`font-semibold ${trainingStatus.status === 'completed' ? 'text-green-600' : 'text-orange-500'}`}>
                  {trainingStatus.status || 'Not Started'} ({trainingStatus.progress || 0}%)
                </span>
              </span>
            </div>
          )}
          <button
            onClick={() => setShowDebugger(true)}
            className="px-5 py-2 bg-indigo-600 text-white rounded-lg shadow hover:bg-indigo-700 transition-colors flex items-center gap-2"
          >
            <Code className="w-5 h-5" />
            Open Debugger
          </button>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel: Data Upload & Configuration */}
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
              <Upload className="w-7 h-7 text-blue-600" /> Upload Your Data
            </h2>
            <p className="text-gray-600">Provide documents and social media exports to train your AI clone.</p>

            <FileUploadArea
              onFilesSelect={handleFilesSelect}
              selectedFiles={selectedFiles}
              onRemoveFile={handleRemoveFile}
              isUploading={isUploading}
            />

            <button
              onClick={handleUploadFiles}
              disabled={selectedFiles.length === 0 || isUploading || !userId}
              className="w-full px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUploading ? <Loader2 className="animate-spin" /> : <FileUp />}
              {isUploading ? 'Uploading Files...' : 'Upload Selected Files'}
            </button>

            {/* Training Settings */}
            <div className="bg-gray-50 rounded-xl p-6 shadow-inner border border-gray-100 space-y-4">
              <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                <Settings className="w-6 h-6 text-purple-600" /> Training Settings
              </h3>
              <div>
                <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-1">
                  Base Model
                </label>
                <select
                  id="model-select"
                  value={metadata.modelName}
                  onChange={(e) => setMetadata({ ...metadata, modelName: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                >
                  {AVAILABLE_MODELS.map((model) => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="task-type-select" className="block text-sm font-medium text-gray-700 mb-1">
                  Task Type
                </label>
                <select
                  id="task-type-select"
                  value={metadata.taskType}
                  onChange={(e) => setMetadata({ ...metadata, taskType: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                >
                  {TASK_TYPES.map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={startTraining}
                disabled={trainingStatus.status === 'in_progress' || trainingStatus.status === 'completed' || !userId}
                className="w-full px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {trainingStatus.status === 'in_progress' ? <Loader2 className="animate-spin" /> : <Play />}
                {trainingStatus.status === 'in_progress' ? 'Training in Progress...' : 'Start Training'}
              </button>
            </div>
          </div>

          {/* Right Panel: AI Chat */}
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
              <MessageSquare className="w-7 h-7 text-green-600" /> Chat with Your Clone
            </h2>
            <p className="text-gray-600">Test your personalized AI model here.</p>

            <div className="bg-gray-50 rounded-xl p-6 shadow-inner border border-gray-100 h-96 flex flex-col">
              <div className="flex-grow overflow-y-auto mb-4 space-y-3 p-2 custom-scrollbar">
                {chatHistory.length === 0 && (
                  <div className="text-center text-gray-500 py-12">
                    <p>Start a conversation with your AI clone!</p>
                    <p className="text-sm">Once trained, it will respond based on your data.</p>
                  </div>
                )}
                {chatHistory.map((msg, index) => (
                  <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`p-3 rounded-lg max-w-[80%] ${
                      msg.sender === 'user'
                        ? 'bg-blue-500 text-white rounded-br-none'
                        : 'bg-gray-200 text-gray-800 rounded-bl-none'
                    }`}>
                      <p className="font-medium">{msg.text}</p>
                      {msg.context && (
                        <p className="text-xs text-gray-600 mt-1 italic">
                          Context: {msg.context}
                        </p>
                      )}
                      {msg.sources && msg.sources.length > 0 && (
                          <p className="text-xs text-gray-600 mt-1">
                              Sources: {msg.sources.join(', ')}
                          </p>
                      )}
                      {msg.score !== null && msg.score !== undefined && (
                          <p className="text-xs text-gray-600 mt-1">
                              Similarity Score: {msg.score.toFixed(4)}
                          </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="border-t border-gray-200 pt-4">
                <textarea
                  className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-blue-500 focus:border-blue-500"
                  rows="3"
                  placeholder="Type your message here..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  disabled={trainingStatus.status !== 'completed' || !userId}
                ></textarea>
                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="openai-polish"
                      checked={useOpenAIPolish}
                      onChange={(e) => setUseOpenAIPolish(e.target.checked)}
                      className="form-checkbox h-4 w-4 text-indigo-600 transition duration-150 ease-in-out rounded"
                      disabled={trainingStatus.status !== 'completed' || !userId}
                    />
                    <label htmlFor="openai-polish" className="text-sm text-gray-700">
                      Use OpenAI for Polishing
                    </label>
                  </div>
                  <button
                    onClick={handleSendMessage}
                    disabled={trainingStatus.status !== 'completed' || !message.trim() || !userId}
                    className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Send className="w-5 h-5" /> Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm mt-12">
          <p>© 2025 AI Training Platform • Personalize Your AI Experience</p>
        </div>
      </div>

      {/* Debugger Modal */}
      {showDebugger && (
        <AIDebugger
          trainingStatus={trainingStatus}
          selectedFiles={selectedFiles}
          metadata={metadata}
          chatHistory={chatHistory}
          userId={userId}
          onClose={() => setShowDebugger(false)}
        />
      )}
    </div>
  );
};

export default AITrainingPlatform;