/**
 * @class ApiService
 * @description Centralized service for all API calls to the backend.
 */
class ApiService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const responseText = await response.text();

      if (!response.ok) {
        let errorData = {};
        try {
          errorData = JSON.parse(responseText);
        } catch (e) {
          errorData = { detail: `HTTP ${response.status}: ${response.statusText}` };
        }
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return responseText ? JSON.parse(responseText) : {};
    } catch (error) {
      console.error(`API request failed for endpoint: ${endpoint}`, error);
      throw error;
    }
  }

  async createUser() {
    return this.request('/api/users', { method: 'POST' });
  }

  async getUserData(userId) {
    return this.request(`/api/users/${userId}/data`);
  }

  async uploadFile(userId, file, trainingConfig) {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('file', file);
    formData.append('training_config', JSON.stringify(trainingConfig));

    return this.request('/api/upload', {
      method: 'POST',
      headers: {}, // don't override headers for FormData
      body: formData,
    });
  }

  /**
   * ✅ FIXED: Sends JSON body to match FastAPI's expected request model,
   * including the selected model_name.
   */
  async startTraining(userId, modelName) { // <--- Added modelName parameter
    return this.request('/api/training/start', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // <--- Included model_name in the JSON body
      body: JSON.stringify({ user_id: userId, model_name: modelName }), 
    });
  }

  async getTrainingStatus(userId) {
    return this.request(`/api/training/status/${userId}`);
  }

  async sendChatMessage(userId, message, useApi = null) {
    return this.request('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        message,
        use_api: useApi,
      }),
    });
  }

  async checkHealth() {
    return this.request('/health');
  }

  async getAvailableModels() {
    return this.request('/api/models');
  }
}

export default new ApiService();
