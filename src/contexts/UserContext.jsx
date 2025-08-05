import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithCustomToken, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, doc, setDoc, getDoc, onSnapshot } from 'firebase/firestore';
import { Loader2, User, WifiOff, FileText, CheckCircle } from 'lucide-react';

// Initialize Firebase configuration from the Canvas environment.
// This is required for the application to connect to the backend services.
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_firebaseConfig) : {};
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';

// Initialize the Firebase app
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

// Use a custom auth token if available, otherwise sign in anonymously.
const initializeAuth = async () => {
  const token = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;
  try {
    if (token) {
      await signInWithCustomToken(auth, token);
    } else {
      await signInAnonymously(auth);
    }
    console.log("Firebase authentication successful.");
  } catch (error) {
    console.error("Firebase authentication failed:", error);
    throw error;
  }
};

const UserContext = createContext();

// Custom hook to use the user context
export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

// Provider component to wrap the application
export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);

  // A memoized function to get the Firestore document path for the user
  const getUserDocRef = useCallback((userId) => {
    return doc(db, 'artifacts', appId, 'users', userId, 'userData', 'profile');
  }, []);

  // Effect to handle Firebase authentication and set up the auth listener
  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, async (authUser) => {
      if (authUser) {
        const userId = authUser.uid;
        console.log("User authenticated with ID:", userId);
        
        const userDocRef = getUserDocRef(userId);
        
        // Listen for real-time updates to the user's data in Firestore
        const unsubscribeSnapshot = onSnapshot(userDocRef, (docSnap) => {
          if (docSnap.exists()) {
            const userData = docSnap.data();
            console.log("User data loaded from Firestore:", userData);
            setUser({ userId, ...userData });
          } else {
            // If the user document doesn't exist, create it with a default profile
            console.log("No user profile found, creating a new one.");
            const defaultProfile = {
              has_trained_model: false,
              training_progress: 0,
              conversation_count: 0,
              data_available: false,
            };
            setDoc(userDocRef, defaultProfile)
              .then(() => {
                console.log("Default profile created successfully.");
                setUser({ userId, ...defaultProfile });
              })
              .catch(error => {
                console.error("Error creating default profile:", error);
                setError("Failed to create user profile.");
              });
          }
          setLoading(false);
        }, (err) => {
          console.error("Firestore snapshot listener failed:", err);
          setError("Failed to listen for user data updates.");
          setLoading(false);
        });
        
        // Return a cleanup function for the snapshot listener
        return () => unsubscribeSnapshot();
      } else {
        // No user is signed in
        console.log("No user signed in.");
        setUser(null);
        setLoading(false);
      }
      setIsAuthReady(true);
    }, (err) => {
      console.error("onAuthStateChanged failed:", err);
      setError("Failed to check authentication state.");
      setLoading(false);
      setIsAuthReady(true);
    });

    // Initial sign-in attempt
    const setupAuth = async () => {
        try {
            await initializeAuth();
        } catch (err) {
            setError("Initial authentication failed.");
            setLoading(false);
            setIsAuthReady(true);
        }
    };
    
    // Only call setupAuth if the auth listener has not been set up yet.
    if (!isAuthReady) {
        setupAuth();
    }
    
    // Cleanup function for the auth listener
    return () => {
        if (unsubscribeAuth) unsubscribeAuth();
    };
  }, [isAuthReady, getUserDocRef]);

  // Function to refresh or update user data in Firestore
  const updateUserData = useCallback(async (updates) => {
    if (!user || !user.userId) {
      setError("Cannot update data: User not authenticated.");
      return;
    }
    setLoading(true);
    try {
      const userDocRef = getUserDocRef(user.userId);
      await setDoc(userDocRef, updates, { merge: true });
      console.log("User data updated successfully.");
      setError(null);
    } catch (err) {
      console.error("Failed to update user data:", err);
      setError("Failed to update user data.");
    } finally {
      setLoading(false);
    }
  }, [user, getUserDocRef]);
  
  const clearError = () => setError(null);

  const value = {
    user,
    loading,
    error,
    updateUserData,
    clearError,
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
};

// Main App component to demonstrate the UserProvider and useUser hook
const App = () => {
  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans antialiased text-slate-800 flex items-center justify-center">
      <div className="w-full max-w-2xl">
        <h1 className="text-4xl font-bold text-center mb-10 text-indigo-700">Firebase User Context Demo</h1>
        <UserProvider>
          <UserProfileCard />
        </UserProvider>
      </div>
    </div>
  );
};

const UserProfileCard = () => {
  const { user, loading, error, updateUserData } = useUser();
  const [isUpdating, setIsUpdating] = useState(false);

  const handleUpdate = async () => {
    setIsUpdating(true);
    // Create some mock data to update
    const mockUpdates = {
      has_trained_model: true,
      training_progress: Math.floor(Math.random() * 100),
      conversation_count: user.conversation_count + 1,
      data_available: Math.random() > 0.5,
      lastUpdated: new Date().toISOString()
    };
    await updateUserData(mockUpdates);
    setIsUpdating(false);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-white rounded-xl shadow-lg border border-slate-200">
        <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
        <p className="mt-4 text-xl font-medium">Loading user data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center p-8 bg-red-50 rounded-xl shadow-lg border border-red-200">
        <WifiOff className="w-12 h-12 text-red-500" />
        <p className="mt-4 text-xl font-medium text-red-700">Error</p>
        <p className="text-red-500 mt-2">{error}</p>
      </div>
    );
  }
  
  if (!user) {
      return (
          <div className="flex flex-col items-center p-8 bg-white rounded-xl shadow-lg border border-slate-200">
              <User className="w-12 h-12 text-gray-400" />
              <p className="mt-4 text-xl font-medium text-gray-600">No user data available.</p>
          </div>
      );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 border border-slate-200">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-slate-800 flex items-center">
          <User className="w-6 h-6 mr-2 text-indigo-600" />
          User Profile
        </h2>
        <span className="text-xs text-slate-500">App ID: {appId}</span>
      </div>
      <div className="space-y-4">
        <div className="flex justify-between items-center bg-slate-50 p-4 rounded-lg">
          <span className="font-medium text-slate-600">User ID:</span>
          <span className="font-mono text-xs text-slate-800 break-all">{user.userId}</span>
        </div>
        <div className="flex justify-between items-center bg-slate-50 p-4 rounded-lg">
          <span className="font-medium text-slate-600">Model Trained:</span>
          {user.has_trained_model ? (
            <span className="text-green-500 font-bold flex items-center">
              <CheckCircle className="w-4 h-4 mr-1" /> Yes
            </span>
          ) : (
            <span className="text-gray-500">No</span>
          )}
        </div>
        <div className="flex justify-between items-center bg-slate-50 p-4 rounded-lg">
          <span className="font-medium text-slate-600">Training Progress:</span>
          <span className="font-bold text-indigo-500">{user.training_progress}%</span>
        </div>
        <div className="flex justify-between items-center bg-slate-50 p-4 rounded-lg">
          <span className="font-medium text-slate-600">Conversation Count:</span>
          <span className="font-bold text-indigo-500">{user.conversation_count}</span>
        </div>
        <div className="flex justify-between items-center bg-slate-50 p-4 rounded-lg">
          <span className="font-medium text-slate-600">Data Available:</span>
          {user.data_available ? (
            <span className="text-green-500 font-bold flex items-center">
              <CheckCircle className="w-4 h-4 mr-1" /> Yes
            </span>
          ) : (
            <span className="text-gray-500">No</span>
          )}
        </div>
      </div>
      <div className="mt-8 flex justify-center">
        <button
          onClick={handleUpdate}
          disabled={isUpdating || loading}
          className="w-full sm:w-auto px-6 py-3 bg-indigo-600 text-white font-semibold rounded-full shadow-md hover:bg-indigo-700 transition duration-300 disabled:bg-indigo-300 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {isUpdating ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Updating...
            </>
          ) : (
            <>
              <FileText className="w-5 h-5 mr-2" />
              Update User Data
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default App;

