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

// Mock data to simulate the platform functionality
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
const FileUploadArea = ({ onFileSelect, selectedFile, onRemoveFile, isUploading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileValidation(files[0]);
    }
  }, []);

  const handleFileValidation = (file) => {
    setUploadError(null);
    
    const validTypes = ['text/plain', 'text/csv', 'application/json', 'application/pdf'];
    const validExtensions = ['.txt', '.csv', '.json', '.pdf'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (file.size > maxSize) {
      setUploadError('File size should be less than 100MB');
      return;
    }
    
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    const isValidType = validTypes.includes(file.type) || validExtensions.includes(fileExtension);
    
    if (!isValidType) {
      setUploadError('Please upload a valid file type: TXT, CSV, JSON, or PDF');
      return;
    }
    
    onFileSelect(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileValidation(file);
    }
  };

  const getFileIcon = (file) => {
    const extension = file.name.split('.').pop().toLowerCase();
    switch (extension) {
      case 'csv': return <Database className="w-8 h-8 text-green-500" />;
      case 'json': return <Code className="w-8 h-8 text-blue-500" />;
      case 'txt': return <FileText className="w-8 h-8 text-gray-500" />;
      case 'pdf': return <FileImage className="w-8 h-8 text-red-500" />;
      default: return <FileText className="w-8 h-8 text-gray-500" />;
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (selectedFile) {
    return (
      <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Selected Dataset</h3>
          <button
            onClick={onRemoveFile}
            disabled={isUploading}
            className="text-gray-400 hover:text-red-500 transition-colors disabled:cursor-not-allowed"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
          {isUploading ? (
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          ) : (
            getFileIcon(selectedFile)
          )}
          <div className="flex-1">
            <h4 className="font-medium text-gray-900">{selectedFile.name}</h4>
            <p className="text-sm text-gray-500 mt-1">
              {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Unknown type'}
            </p>
            {isUploading && (
              <p className="text-sm text-blue-600 mt-1">Processing...</p>
            )}
          </div>
          {!isUploading && <CheckCircle className="w-6 h-6 text-green-500" />}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Dataset</h3>
      
      {uploadError && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">{uploadError}</p>
        </div>
      )}
      
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer ${
          isDragOver 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <FileUp className={`w-12 h-12 mx-auto mb-4 ${isDragOver ? 'text-blue-500' : 'text-gray-400'}`} />
        <h4 className="text-lg font-medium text-gray-900 mb-2">
          {isDragOver ? 'Drop your file here' : 'Upload your dataset'}
        </h4>
        <p className="text-gray-500 mb-4">
          Drag and drop your file here, or click to browse
        </p>
        <p className="text-sm text-gray-400">
          Supported formats: TXT, CSV, JSON, PDF • Max size: 100MB
        </p>
        
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileChange}
          className="hidden"
          accept=".txt,.csv,.json,.pdf,text/plain,text/csv,application/json,application/pdf"
        />
      </div>
    </div>
  );
};

// Training Status Component
const TrainingStatus = ({ status, progress, message, onStart, onStop, isStartEnabled, metrics, usedModel }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'pending': return 'text-gray-500';
      case 'fetching': 
      case 'preprocessing': 
      case 'training': return 'text-blue-500';
      case 'completed': return 'text-green-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'fetching':
      case 'preprocessing':
      case 'training': return <Brain className="w-5 h-5 animate-pulse" />;
      case 'completed': return <CheckCircle className="w-5 h-5" />;
      case 'error': return <AlertCircle className="w-5 h-5" />;
      default: return <Brain className="w-5 h-5" />;
    }
  };

  const isTraining = ['fetching', 'preprocessing', 'training'].includes(status);

  return (
    <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-6 h-6 text-blue-500" />
          <h3 className="text-lg font-semibold text-gray-900">Training Status</h3>
        </div>
        
        <div className={`flex items-center gap-2 ${getStatusColor()}`}>
          {getStatusIcon()}
          <span className="font-medium capitalize">{status}</span>
        </div>
      </div>

      {/* Progress Bar */}
      {isTraining && (
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700">Training Progress</span>
            <span className="text-sm text-gray-500">{progress?.toFixed(1) || 0}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress || 0}%` }}
            />
          </div>
          {message && (
            <p className="text-xs text-gray-500 mt-2">{message}</p>
          )}
        </div>
      )}

      {/* Status Message */}
      {message && !isTraining && (
        <div className="mb-6 p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-700">{message}</p>
        </div>
      )}

      {/* Used Model Display - Prominently Featured */}
      {usedModel && status === 'completed' && (
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
          <div className="flex items-center gap-2 mb-2">
            <Star className="w-5 h-5 text-green-600" />
            <h4 className="font-semibold text-green-800">Your Trained Model</h4>
          </div>
          <div className="text-green-700 text-lg font-mono bg-white px-3 py-2 rounded border">
            {usedModel}
          </div>
          <p className="text-sm text-green-600 mt-2">This is the model that learned from your data!</p>
        </div>
      )}

      {/* Control Buttons */}
      <div className="flex gap-3">
        <button
          onClick={onStart}
          disabled={!isStartEnabled || isTraining}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-400 text-white font-medium rounded-lg transition-all duration-200 disabled:cursor-not-allowed"
        >
          {isTraining ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {isTraining ? 'Training...' : 'Start Training'}
        </button>

        <button
          onClick={onStop}
          disabled={status === 'pending' && !isTraining} 
          className="flex items-center gap-2 px-6 py-3 bg-red-500 hover:bg-red-600 disabled:bg-gray-300 text-white font-medium rounded-lg transition-all duration-200 disabled:cursor-not-allowed"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Training Metrics */}
      {metrics && (
        <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Data Points</p>
            <p className="text-lg font-semibold text-gray-900">
              {metrics.training_data_size || 1250}
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Accuracy</p>
            <p className="text-lg font-semibold text-gray-900">
              {metrics.accuracy || '94.2%'}
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Epochs</p>
            <p className="text-lg font-semibold text-gray-900">
              {metrics.epochs || 15}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

// Chatbot Component
const Chatbot = ({ userId, onSendMessage, chatHistory, isLoadingResponse, onReset, usedModel }) => {
  const [message, setMessage] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleSend = () => {
    if (message.trim() && !isLoadingResponse) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="bg-white border-2 border-gray-200 rounded-xl p-6 flex flex-col h-[600px] max-w-4xl mx-auto shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <MessageSquare className="w-6 h-6 text-purple-500" />
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Chat with your AI Clone</h3>
            {usedModel && (
              <p className="text-sm text-purple-600 font-mono">Powered by: {usedModel}</p>
            )}
          </div>
        </div>
        <button
          onClick={onReset}
          className="flex items-center gap-1 text-sm text-red-500 hover:text-red-700 transition-colors"
        >
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto pr-2 mb-4 space-y-4">
        {chatHistory.length === 0 && (
          <div className="text-center text-gray-500 py-10">
            <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-lg mb-2">Start a conversation with your trained AI!</p>
            <p className="text-sm">Your model has learned from your data and is ready to chat.</p>
          </div>
        )}
        {chatHistory.map((msg, index) => (
          <div 
            key={index} 
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div 
              className={`max-w-[70%] p-4 rounded-lg shadow-sm ${
                msg.sender === 'user' 
                  ? 'bg-blue-500 text-white rounded-br-none' 
                  : 'bg-gray-100 text-gray-800 rounded-bl-none border'
              }`}
            >
              <p className="font-medium text-sm mb-1 flex items-center gap-2">
                {msg.sender === 'user' ? (
                  <>
                    <User className="w-4 h-4" />
                    You
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    AI Clone
                  </>
                )}
              </p>
              <p className="leading-relaxed">{msg.message}</p>
            </div>
          </div>
        ))}
        {isLoadingResponse && (
          <div className="flex justify-start">
            <div className="max-w-[70%] p-4 rounded-lg shadow-sm bg-gray-100 text-gray-800 rounded-bl-none border">
              <p className="font-medium text-sm mb-1 flex items-center gap-2">
                <Brain className="w-4 h-4" />
                AI Clone
              </p>
              <div className="flex items-center gap-2">
                <Loader2 className="w-5 h-5 text-gray-500 animate-spin" />
                <span className="text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Message Input */}
      <div className="flex items-center gap-3 border-t border-gray-200 pt-4">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask your AI about anything from your training data..."
          rows={1}
          className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
          disabled={isLoadingResponse}
        />
        <button
          onClick={handleSend}
          disabled={!message.trim() || isLoadingResponse}
          className="p-3 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

// Main Application Component
const AITrainingPlatform = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [userId] = useState('demo-user-' + Math.random().toString(36).substr(2, 9));
  const [metadata, setMetadata] = useState({
    modelName: '',
    taskType: '',
    version: '1.0.0',
    learningRate: '0.0001',
    description: ''
  });
  const [trainingStatus, setTrainingStatus] = useState({
    status: 'pending',
    progress: 0,
    message: 'Ready to start training',
    metrics: null
  });
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoadingChatResponse, setIsLoadingChatResponse] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [usedModel, setUsedModel] = useState(null); // Track the model used in training

  // Simulate file upload
  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setUploadError(null);
    setIsUploading(true);

    // Simulate upload delay
    setTimeout(() => {
      setIsUploading(false);
      setTrainingStatus(prev => ({
        ...prev,
        message: 'File uploaded successfully. Ready to start training.'
      }));
    }, 2000);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadError(null);
  };

  // Simulate training process
  const startTraining = async () => {
    if (!selectedFile || !metadata.modelName || !metadata.taskType) {
      setUploadError('Please select a file and configure your model first.');
      return;
    }

    setUsedModel(metadata.modelName); // Store the selected model
    setTrainingStatus({ status: 'preprocessing', progress: 10, message: 'Preprocessing your data...' });

    // Simulate training steps
    const steps = [
      { status: 'preprocessing', progress: 25, message: 'Analyzing your dataset...', delay: 2000 },
      { status: 'training', progress: 50, message: `Training ${metadata.modelName} model...`, delay: 3000 },
      { status: 'training', progress: 75, message: 'Fine-tuning parameters...', delay: 2000 },
      { status: 'training', progress: 90, message: 'Validating model performance...', delay: 1500 },
      { 
        status: 'completed', 
        progress: 100, 
        message: `Training completed! Your ${metadata.modelName} model is ready.`,
        metrics: {
          training_data_size: 1250,
          accuracy: '94.2%',
          epochs: 15
        },
        delay: 1000 
      }
    ];

    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, step.delay));
      setTrainingStatus(step);
    }

    // Show chatbot after completion
    setTimeout(() => {
      setIsChatting(true);
    }, 1500);
  };

  const resetTraining = () => {
    setTrainingStatus({
      status: 'pending',
      progress: 0,
      message: 'Ready to start training',
      metrics: null
    });
    setSelectedFile(null);
    setUploadError(null);
    setChatHistory([]);
    setIsChatting(false);
    setIsLoadingChatResponse(false);
    setUsedModel(null);
  };

  // Simulate chat responses
  const handleSendMessage = async (message) => {
    setChatHistory(prev => [...prev, { sender: 'user', message }]);
    setIsLoadingChatResponse(true);

    // Simulate AI response delay
    setTimeout(() => {
      const responses = [
        `Based on your training data, I understand that ${message.toLowerCase()} is an interesting topic. Let me share my thoughts...`,
        `From what I learned during training, I can tell you that this relates to the patterns I observed in your dataset.`,
        `That's a great question! My training on your specific data helps me provide personalized insights about this.`,
        `I processed this type of information during my training phase. Here's what I found most relevant...`,
        `Your training data contained similar themes. I can provide a response tailored to your specific context.`
      ];
      
      const response = responses[Math.floor(Math.random() * responses.length)];
      setChatHistory(prev => [...prev, { sender: 'bot', message: response }]);
      setIsLoadingChatResponse(false);
    }, 1500);
  };

  const isStartEnabled = selectedFile && metadata.modelName && metadata.taskType && !isUploading;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 p-4 font-sans">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Training Platform
            </h1>
          </div>
          <p className="text-gray-600 max-w-3xl mx-auto mb-4 text-lg">
            Transform your data into intelligent AI models. Upload your dataset, configure parameters, and create personalized AI assistants that understand your unique context.
          </p>
          
          {/* Feature Highlights */}
          <div className="flex flex-wrap justify-center gap-6 mb-6">
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-blue-200 text-sm">
              <Zap className="w-4 h-4 text-blue-500" />
              <span>Real-time Training</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-green-200 text-sm">
              <Target className="w-4 h-4 text-green-500" />
              <span>Personalized Models</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-purple-200 text-sm">
              <Activity className="w-4 h-4 text-purple-500" />
              <span>Interactive Chat</span>
            </div>
          </div>
          
          {/* User Info */}
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-lg border border-gray-200 text-sm text-gray-600 shadow-sm">
            <User className="w-4 h-4" />
            <span>Session ID: {userId}</span>
            {usedModel && (
              <span className="text-green-600 font-medium">• Model: {usedModel}</span>
            )}
          </div>
        </div>

        {/* Error Display */}
        {uploadError && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg max-w-4xl mx-auto">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-700">{uploadError}</p>
              <button 
                onClick={() => setUploadError(null)}
                className="ml-auto text-red-500 hover:text-red-700"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Main Content */}
        {isChatting ? (
          <Chatbot 
            userId={userId}
            onSendMessage={handleSendMessage}
            chatHistory={chatHistory}
            isLoadingResponse={isLoadingChatResponse}
            onReset={resetTraining}
            usedModel={usedModel}
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Left Column - Upload & Configuration */}
            <div className="space-y-8">
              <FileUploadArea
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                onRemoveFile={handleRemoveFile}
                isUploading={isUploading}
              />
              
              {/* Model Configuration */}
              <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                  <Settings className="w-6 h-6 text-blue-500" />
                  <h3 className="text-lg font-semibold text-gray-900">Model Configuration</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Base Model *
                    </label>
                    <select
                      value={metadata.modelName}
                      onChange={(e) => setMetadata({ ...metadata, modelName: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    >
                      <option value="">Select a model</option>
                      {AVAILABLE_MODELS.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Task Type *
                    </label>
                    <select
                      value={metadata.taskType}
                      onChange={(e) => setMetadata({ ...metadata, taskType: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    >
                      <option value="">Select task type</option>
                      {TASK_TYPES.map(task => (
                        <option key={task} value={task}>
                          {task.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Learning Rate
                    </label>
                    <input
                      type="number"
                      step="0.0001"
                      value={metadata.learningRate}
                      onChange={(e) => setMetadata({ ...metadata, learningRate: e.target.value })}
                      placeholder="0.0001"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Version
                    </label>
                    <input
                      type="text"
                      value={metadata.version}
                      onChange={(e) => setMetadata({ ...metadata, version: e.target.value })}
                      placeholder="e.g., v1.0.0"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    />
                  </div>
                </div>

                <div className="mt-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={metadata.description}
                    onChange={(e) => setMetadata({ ...metadata, description: e.target.value })}
                    placeholder="Describe your model and training objectives..."
                    rows={3}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 resize-none"
                  />
                </div>
              </div>
            </div>

            {/* Right Column - Training Status */}
            <div className="space-y-8">
              <TrainingStatus
                status={trainingStatus.status}
                progress={trainingStatus.progress}
                message={trainingStatus.message}
                metrics={trainingStatus.metrics}
                onStart={startTraining}
                onStop={resetTraining}
                isStartEnabled={isStartEnabled}
                usedModel={usedModel}
              />
              
              {/* Dataset Preview */}
              <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Preview</h3>
                {selectedFile ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
                      <FileText className="w-8 h-8 text-blue-500" />
                      <div>
                        <h4 className="font-medium text-gray-900">{selectedFile.name}</h4>
                        <p className="text-sm text-gray-500">
                          {(selectedFile.size / 1024).toFixed(1)} KB • Ready for training
                        </p>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 space-y-1">
                      <p className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        File uploaded successfully
                      </p>
                      <p className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Format validated
                      </p>
                      <p className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Ready for model training
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                    <p>Upload a dataset to see preview</p>
                    <p className="text-sm mt-2">Your data will be processed and validated here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm mt-12">
          <p>© 2025 AI Training Platform • Personalize Your AI Experience</p>
        </div>
      </div>
    </div>
  );
};

export default AITrainingPlatform;