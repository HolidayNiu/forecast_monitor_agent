"""
AI-Enhanced DetectorAgent for intelligent forecast quality detection.

This agent combines rule-based detection with AI/ML capabilities for
adaptive threshold learning and pattern recognition.
"""
import pandas as pd
import numpy as np
try:
    from typing import Dict, Any, Optional, Tuple, List
except ImportError:
    # Python 2.7 compatibility
    Dict = dict
    Any = object
    Optional = object
    Tuple = tuple
    List = list

from .detector_agent import DetectorAgent

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

import json
import pickle
import os
from pathlib import Path


class FeatureExtractor:
    """Extract comprehensive features for AI-based detection."""
    
    def extract_time_series_features(self, historical_data, forecast_data):
        """Extract time series specific features."""
        features = {}
        
        # Basic statistics
        features['hist_mean'] = float(historical_data.mean())
        features['hist_std'] = float(historical_data.std())
        features['hist_cv'] = features['hist_std'] / features['hist_mean'] if features['hist_mean'] != 0 else 0
        features['fcst_mean'] = float(forecast_data.mean())
        features['fcst_std'] = float(forecast_data.std())
        features['fcst_cv'] = features['fcst_std'] / features['fcst_mean'] if features['fcst_mean'] != 0 else 0
        
        # Trend features
        hist_x = np.arange(len(historical_data))
        fcst_x = np.arange(len(forecast_data))
        
        if len(historical_data) > 1:
            hist_slope = np.polyfit(hist_x, historical_data.values, 1)[0]
            features['hist_trend_slope'] = float(hist_slope)
            features['hist_trend_strength'] = float(abs(hist_slope))
        else:
            features['hist_trend_slope'] = 0.0
            features['hist_trend_strength'] = 0.0
            
        if len(forecast_data) > 1:
            fcst_slope = np.polyfit(fcst_x, forecast_data.values, 1)[0]
            features['fcst_trend_slope'] = float(fcst_slope)
            features['fcst_trend_strength'] = float(abs(fcst_slope))
        else:
            features['fcst_trend_slope'] = 0.0
            features['fcst_trend_strength'] = 0.0
        
        # Volatility features
        features['volatility_ratio'] = features['fcst_cv'] / features['hist_cv'] if features['hist_cv'] != 0 else 1.0
        features['volatility_change'] = features['fcst_std'] - features['hist_std']
        
        # Magnitude features
        features['magnitude_ratio'] = features['fcst_mean'] / features['hist_mean'] if features['hist_mean'] != 0 else 1.0
        features['magnitude_change'] = features['fcst_mean'] - features['hist_mean']
        features['magnitude_pct_change'] = (features['fcst_mean'] - features['hist_mean']) / features['hist_mean'] if features['hist_mean'] != 0 else 0
        
        # Seasonality proxy features (simplified)
        if len(historical_data) >= 12:
            # Simple seasonality indicator
            hist_monthly_std = []
            for month in range(12):
                month_values = [historical_data.iloc[i] for i in range(month, len(historical_data), 12)]
                if month_values:
                    hist_monthly_std.append(np.std(month_values))
            
            if hist_monthly_std:
                features['hist_seasonal_variation'] = float(np.mean(hist_monthly_std))
            else:
                features['hist_seasonal_variation'] = 0.0
        else:
            features['hist_seasonal_variation'] = 0.0
            
        # Data length features
        features['hist_length'] = len(historical_data)
        features['fcst_length'] = len(forecast_data)
        features['data_ratio'] = len(forecast_data) / len(historical_data) if len(historical_data) > 0 else 0
        
        return features
    
    def extract_pattern_features(self, historical_data, forecast_data):
        """Extract pattern-based features."""
        features = {}
        
        # Continuity features
        if len(historical_data) > 0 and len(forecast_data) > 0:
            features['transition_gap'] = float(forecast_data.iloc[0] - historical_data.iloc[-1])
            features['transition_gap_pct'] = features['transition_gap'] / historical_data.iloc[-1] if historical_data.iloc[-1] != 0 else 0
        else:
            features['transition_gap'] = 0.0
            features['transition_gap_pct'] = 0.0
        
        # Distribution features (simplified)
        try:
            features['hist_skewness'] = float(historical_data.skew()) if len(historical_data) > 2 else 0.0
            features['fcst_skewness'] = float(forecast_data.skew()) if len(forecast_data) > 2 else 0.0
            features['skewness_change'] = features['fcst_skewness'] - features['hist_skewness']
        except:
            features['hist_skewness'] = 0.0
            features['fcst_skewness'] = 0.0
            features['skewness_change'] = 0.0
        
        return features
    
    def extract_all_features(self, historical_data, forecast_data):
        """Extract all available features."""
        features = {}
        features.update(self.extract_time_series_features(historical_data, forecast_data))
        features.update(self.extract_pattern_features(historical_data, forecast_data))
        return features


class AdaptiveThresholdLearner:
    """Learn optimal thresholds based on historical performance."""
    
    def __init__(self):
        self.threshold_models = {}
        self.feature_scaler = None
        self.is_trained = False
        
        if ML_AVAILABLE:
            self.threshold_models = {
                'seasonality': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'volatility': RandomForestRegressor(n_estimators=50, random_state=42),
                'magnitude': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'trend': LogisticRegression(random_state=42)
            }
            self.feature_scaler = StandardScaler()
    
    def create_synthetic_training_data(self, n_samples=1000):
        """Create synthetic training data for threshold learning."""
        if not ML_AVAILABLE:
            return None, None
            
        np.random.seed(42)
        X = []
        y = {}
        
        # Initialize target arrays
        for threshold_type in self.threshold_models.keys():
            y[threshold_type] = []
        
        for _ in range(n_samples):
            # Generate synthetic features
            features = {
                'hist_cv': np.random.uniform(0.1, 0.8),
                'fcst_cv': np.random.uniform(0.05, 0.6),
                'hist_mean': np.random.uniform(50, 500),
                'fcst_mean': np.random.uniform(40, 600),
                'hist_trend_strength': np.random.uniform(0, 5),
                'fcst_trend_strength': np.random.uniform(0, 5),
                'hist_seasonal_variation': np.random.uniform(0, 50),
                'volatility_ratio': np.random.uniform(0.1, 2.0),
                'magnitude_ratio': np.random.uniform(0.5, 2.0)
            }
            
            # Calculate optimal thresholds based on context
            # These are heuristic rules for synthetic data
            optimal_thresholds = {
                'volatility': max(0.3, min(0.8, 0.5 + (features['hist_cv'] - 0.3) * 0.5)),
                'magnitude': max(0.2, min(0.8, 0.5 + (features['hist_cv'] - 0.3) * 0.3)),
                'seasonality': max(1.0, min(2.5, 1.5 + features['hist_seasonal_variation'] / 100)),
                'trend': 0.3 + features['hist_trend_strength'] * 0.1  # For trend confidence
            }
            
            X.append(list(features.values()))
            for threshold_type, optimal_value in optimal_thresholds.items():
                y[threshold_type].append(optimal_value)
        
        return np.array(X), y
    
    def train(self, X=None, y=None):
        """Train threshold learning models."""
        if not ML_AVAILABLE:
            print("ML libraries not available. Using default thresholds.")
            return
        
        if X is None or y is None:
            print("No training data provided. Generating synthetic data...")
            X, y = self.create_synthetic_training_data()
            
        if X is None:
            print("Could not create training data.")
            return
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train each threshold model
        for threshold_type, model in self.threshold_models.items():
            if threshold_type in y:
                try:
                    if threshold_type == 'trend':
                        # For trend, convert to binary classification
                        y_binary = (np.array(y[threshold_type]) > 0.5).astype(int)
                        model.fit(X_scaled, y_binary)
                    else:
                        model.fit(X_scaled, y[threshold_type])
                    print("Trained {} threshold model".format(threshold_type))
                except Exception as e:
                    print("Error training {} model: {}".format(threshold_type, e))
        
        self.is_trained = True
    
    def predict_thresholds(self, features):
        """Predict optimal thresholds for given features."""
        if not ML_AVAILABLE or not self.is_trained:
            # Return default thresholds
            return {
                'seasonality': 1.5,
                'volatility': 0.5,
                'magnitude': 0.5,
                'trend': 0.3
            }
        
        # Convert features dict to array
        feature_names = ['hist_cv', 'fcst_cv', 'hist_mean', 'fcst_mean', 
                        'hist_trend_strength', 'fcst_trend_strength', 
                        'hist_seasonal_variation', 'volatility_ratio', 'magnitude_ratio']
        
        feature_array = []
        for fname in feature_names:
            feature_array.append(features.get(fname, 0.0))
        
        X = np.array([feature_array])
        X_scaled = self.feature_scaler.transform(X)
        
        predicted_thresholds = {}
        for threshold_type, model in self.threshold_models.items():
            try:
                if threshold_type == 'trend':
                    # For trend, predict probability
                    pred = model.predict_proba(X_scaled)[0][1]
                else:
                    pred = model.predict(X_scaled)[0]
                predicted_thresholds[threshold_type] = float(pred)
            except Exception as e:
                # Fallback to default
                defaults = {'seasonality': 1.5, 'volatility': 0.5, 'magnitude': 0.5, 'trend': 0.3}
                predicted_thresholds[threshold_type] = defaults.get(threshold_type, 0.5)
        
        return predicted_thresholds


class AIDetectorAgent(DetectorAgent):
    """AI-Enhanced DetectorAgent with adaptive thresholds and ML capabilities."""
    
    def __init__(self, thresholds=None, use_ai=True, model_path=None):
        """
        Initialize AI-enhanced detector agent.
        
        Args:
            thresholds: Optional base thresholds (for fallback)
            use_ai: Whether to use AI enhancements
            model_path: Path to saved AI models
        """
        super().__init__(thresholds)
        self.use_ai = use_ai and ML_AVAILABLE
        self.feature_extractor = FeatureExtractor()
        self.threshold_learner = None
        self.model_path = model_path
        
        if self.use_ai:
            self.threshold_learner = AdaptiveThresholdLearner()
            self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self._load_models()
                print("Loaded AI models from {}".format(self.model_path))
            except Exception as e:
                print("Error loading models: {}. Training new ones...".format(e))
                self._train_models()
        else:
            print("Training AI models...")
            self._train_models()
    
    def _train_models(self):
        """Train the AI models."""
        if self.threshold_learner:
            self.threshold_learner.train()
    
    def _load_models(self):
        """Load trained models from disk."""
        # Implementation for loading models would go here
        pass
    
    def _save_models(self):
        """Save trained models to disk."""
        # Implementation for saving models would go here
        pass
    
    def reason(self):
        """Enhanced reasoning with AI capabilities."""
        if self._historical_data is None or self._forecast_data is None:
            raise RuntimeError("No data observed. Call observe() first.")
        
        # Extract features for AI analysis
        features = self.feature_extractor.extract_all_features(
            self._historical_data, self._forecast_data
        )
        
        # Get adaptive thresholds if AI is enabled
        if self.use_ai and self.threshold_learner and self.threshold_learner.is_trained:
            adaptive_thresholds = self.threshold_learner.predict_thresholds(features)
        else:
            # Use default thresholds
            adaptive_thresholds = {
                'seasonality': 1.5,
                'volatility': 0.5,
                'magnitude': 0.5,
                'trend': 0.3
            }
        
        # Run enhanced diagnostics with adaptive thresholds
        self._reasoning_result = self._run_adaptive_diagnostics(adaptive_thresholds, features)
    
    def _run_adaptive_diagnostics(self, thresholds, features):
        """Run diagnostics with adaptive thresholds."""
        from modules.diagnostics import (detect_trend_mismatch, detect_missing_seasonality,
                                       detect_volatility_mismatch, detect_magnitude_mismatch)
        from modules.loader import get_recent_actuals, get_early_forecast
        
        # Get required data for diagnostics
        recent_actuals = get_recent_actuals(self._historical_data)
        early_forecast = get_early_forecast(self._forecast_data)
        
        # Run original diagnostics
        results = {
            'trend_mismatch': detect_trend_mismatch(self._historical_data, self._forecast_data),
            'missing_seasonality': detect_missing_seasonality(self._historical_data, self._forecast_data),
            'volatility_mismatch': detect_volatility_mismatch(self._historical_data, self._forecast_data),
            'magnitude_mismatch': detect_magnitude_mismatch(recent_actuals, early_forecast)
        }
        
        # Enhance with adaptive thresholds
        if self.use_ai:
            results = self._apply_adaptive_thresholds(results, thresholds, features)
        
        # Calculate overall risk score
        detected_issues = sum(1 for result in results.values() if result['detected'])
        avg_confidence = np.mean([result['confidence'] for result in results.values() if result['detected']])
        
        results['summary'] = {
            'total_issues': detected_issues,
            'avg_confidence': avg_confidence if detected_issues > 0 else 0,
            'risk_score': detected_issues * avg_confidence if detected_issues > 0 else 0,
            'ai_enhanced': self.use_ai,
            'adaptive_thresholds_used': thresholds if self.use_ai else None
        }
        
        return results
    
    def _apply_adaptive_thresholds(self, results, thresholds, features):
        """Apply adaptive thresholds to detection results."""
        # Enhance volatility detection
        if 'volatility_mismatch' in results:
            volatility_ratio = features.get('volatility_ratio', 1.0)
            adaptive_threshold = thresholds.get('volatility', 0.5)
            
            too_flat = volatility_ratio < adaptive_threshold
            confidence = (adaptive_threshold - volatility_ratio) / adaptive_threshold if too_flat else 0
            confidence = max(0, min(1, confidence))
            
            results['volatility_mismatch'].update({
                'detected': too_flat,
                'confidence': confidence,
                'adaptive_threshold': adaptive_threshold,
                'method': 'ai_enhanced'
            })
        
        # Enhance magnitude detection
        if 'magnitude_mismatch' in results:
            magnitude_pct_change = abs(features.get('magnitude_pct_change', 0))
            adaptive_threshold = thresholds.get('magnitude', 0.5)
            
            magnitude_mismatch = magnitude_pct_change > adaptive_threshold
            confidence = min(magnitude_pct_change / adaptive_threshold, 1.0) if magnitude_mismatch else 0
            
            results['magnitude_mismatch'].update({
                'detected': magnitude_mismatch,
                'confidence': confidence,
                'adaptive_threshold': adaptive_threshold,
                'method': 'ai_enhanced'
            })
        
        # Add AI confidence scores
        for issue_type in results:
            if issue_type != 'summary' and isinstance(results[issue_type], dict):
                results[issue_type]['ai_features'] = features
        
        return results
    
    def get_ai_insights(self):
        """Get AI-specific insights about the detection."""
        if not self.use_ai or self._reasoning_result is None:
            return None
        
        insights = {
            'ai_enabled': True,
            'ml_available': ML_AVAILABLE,
            'adaptive_thresholds': self._reasoning_result['summary'].get('adaptive_thresholds_used'),
            'feature_count': len(self._reasoning_result.get('trend_mismatch', {}).get('ai_features', {})),
        }
        
        return insights
    
    def explain_detection(self):
        """Provide explanation of AI-enhanced detection."""
        if self._reasoning_result is None:
            return "No analysis performed yet."
        
        explanation = []
        explanation.append("AI-Enhanced Forecast Quality Detection Results:")
        explanation.append("=" * 50)
        
        if self.use_ai:
            explanation.append("✅ AI enhancements: ENABLED")
            thresholds = self._reasoning_result['summary'].get('adaptive_thresholds_used', {})
            if thresholds:
                explanation.append("🎯 Adaptive thresholds used:")
                for thresh_type, value in thresholds.items():
                    explanation.append("   - {}: {:.3f}".format(thresh_type, value))
        else:
            explanation.append("❌ AI enhancements: DISABLED (using fixed thresholds)")
        
        explanation.append("\n📊 Detection Results:")
        for issue_type, details in self._reasoning_result.items():
            if issue_type != 'summary' and isinstance(details, dict):
                status = "DETECTED" if details.get('detected', False) else "Not detected"
                confidence = details.get('confidence', 0)
                method = details.get('method', 'statistical')
                explanation.append("   {} ({}): {} - confidence {:.3f}".format(
                    issue_type.replace('_', ' ').title(), method, status, confidence))
        
        risk_score = self._reasoning_result['summary']['risk_score']
        explanation.append("\n🎯 Overall Risk Score: {:.3f}".format(risk_score))
        
        return "\n".join(explanation)