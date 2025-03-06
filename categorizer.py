# categorizer.py
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import shutil
import json
import logging
from typing import List, Dict, Union, Any, Optional, Tuple
import numpy as np
import cv2
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ImageCategorizer:
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14", # Upgraded to larger model
        config_path: str = "domain_config.json",
        confidence_threshold: float = 0.15,
        enable_preprocessing: bool = True,
        enable_learning: bool = True
    ):
        """
        Initialize the Image Categorizer with CLIP model and configuration.
        
        Args:
            model_name: The CLIP model to use (using larger model for better performance)
            config_path: Path to domain configuration file
            confidence_threshold: Minimum confidence required to accept a prediction
            enable_preprocessing: Whether to apply image preprocessing
            enable_learning: Whether to enable feedback-based learning
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        self.enable_learning = enable_learning
        
        # Load domains configuration
        self.domains = self.load_or_create_config()
        
        # Create analytics directory
        self.analytics_dir = "analytics"
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Initialize feedback collection
        self.feedback_file = os.path.join(self.analytics_dir, "feedback.json")
        self.load_or_create_feedback()
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_categorizations": 0,
            "low_confidence_count": 0,
            "domain_counts": {},
            "confidence_scores": [],
            "confusion_matrix": {},  # Will store user corrections
        }
        
        try:
            self.logger.info(f"Loading CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            # Set device to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {e}")
            raise
        
        self.output_base_dir = "categorized_images"
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Create feedback directory for corrected images
        self.feedback_dir = os.path.join(self.output_base_dir, "_feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def load_or_create_feedback(self) -> None:
        """
        Load or create the feedback collection file.
        """
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
                    self.logger.info(f"Loaded {len(self.feedback_data)} feedback entries")
            except Exception as e:
                self.logger.error(f"Error loading feedback data: {e}")
                self.feedback_data = {"entries": [], "corrections": {}}
        else:
            self.feedback_data = {"entries": [], "corrections": {}}
            try:
                with open(self.feedback_file, 'w') as f:
                    json.dump(self.feedback_data, f, indent=4)
            except Exception as e:
                self.logger.error(f"Error creating feedback file: {e}")
    
    def load_or_create_config(self) -> Dict[str, Dict]:
        """
        Load or create domain configuration with more detailed descriptions.
        """
        default_domains = {
            "Technology": {
                "descriptions": [
                    "a screenshot of a computer interface", 
                    "software development environment", 
                    "programming code editor", 
                    "tech dashboard", 
                    "computer settings", 
                    "software application",
                    "tech hardware device",
                    "operating system interface",
                    "developer tools",
                    "artificial intelligence interface",
                    "cybersecurity dashboard",
                    "IoT device control panel",
                    "network management interface"
                ],
                "weight": 1.0
            },
            "Finance": {
                "descriptions": [
                    "financial dashboard", 
                    "banking website", 
                    "stock market interface", 
                    "investment platform", 
                    "online banking screen", 
                    "cryptocurrency exchange",
                    "financial planning tool",
                    "budget tracking application",
                    "payment processing screen",
                    "invoicing system",
                    "tax preparation software",
                    "financial analytics dashboard",
                    "financial news platform"
                ],
                "weight": 1.0
            },
            "Health": {
                "descriptions": [
                    "medical application", 
                    "health tracking screen", 
                    "fitness app", 
                    "medical records interface", 
                    "telemedicine platform", 
                    "health monitoring dashboard",
                    "nutrition tracker",
                    "wellness application",
                    "mental health platform",
                    "medical imaging viewer",
                    "hospital management system",
                    "electronic health records",
                    "health insurance portal"
                ],
                "weight": 1.0
            },
            "Education": {
                "descriptions": [
                    "online learning platform", 
                    "educational website", 
                    "study materials screen", 
                    "course management system", 
                    "online classroom interface", 
                    "educational resource page",
                    "learning management system",
                    "educational quiz platform",
                    "digital textbook",
                    "student dashboard",
                    "academic scheduling tool",
                    "e-learning course interface",
                    "virtual classroom environment"
                ],
                "weight": 1.0
            },
            "Travel": {
                "descriptions": [
                    "travel booking website", 
                    "maps application", 
                    "travel planning interface", 
                    "airline booking screen", 
                    "hotel reservation page", 
                    "navigation application",
                    "vacation packages platform",
                    "travel itinerary planner",
                    "car rental service",
                    "travel guide website",
                    "travel review platform",
                    "transportation booking app",
                    "travel expense tracker"
                ],
                "weight": 1.0
            },
            "Entertainment": {
                "descriptions": [
                    "streaming platform", 
                    "game interface", 
                    "media website", 
                    "video platform", 
                    "entertainment application", 
                    "movie or TV show screen",
                    "music streaming service",
                    "gaming dashboard",
                    "esports platform",
                    "video editing interface",
                    "media player application",
                    "podcast platform",
                    "digital entertainment library"
                ],
                "weight": 1.0
            },
            "E-commerce": {
                "descriptions": [
                    "online shopping website", 
                    "product page", 
                    "marketplace interface", 
                    "shopping cart screen", 
                    "product catalog", 
                    "online store dashboard",
                    "e-commerce checkout page",
                    "online auction website",
                    "digital storefront",
                    "product review page",
                    "order tracking interface",
                    "customer account dashboard",
                    "subscription management page"
                ],
                "weight": 1.0
            },
            "Social Media": {
                "descriptions": [
                    "social network interface", 
                    "messaging platform", 
                    "social media feed", 
                    "chat application", 
                    "profile page", 
                    "social network dashboard",
                    "social media post composer",
                    "timeline view",
                    "friend or connection management",
                    "photo sharing platform",
                    "social media comment section",
                    "direct messaging interface",
                    "social media story creator",
                    "follower analytics dashboard",
                    "content discovery feed",
                    "trending topics section",
                    "live streaming interface"
                ],
                "weight": 1.0
            },
            "News": {
                "descriptions": [
                    "news website", 
                    "online newspaper", 
                    "news article page", 
                    "news feed", 
                    "current events portal", 
                    "media news interface",
                    "breaking news alert",
                    "news aggregator",
                    "digital magazine",
                    "news subscription service",
                    "editorial dashboard",
                    "news video platform",
                    "headline ticker"
                ],
                "weight": 1.0
            },
            "Productivity": {
                "descriptions": [
                    "task management app", 
                    "project tracking interface", 
                    "productivity tool", 
                    "to-do list", 
                    "workflow management", 
                    "collaboration platform",
                    "calendar application",
                    "note-taking software",
                    "time tracking dashboard",
                    "document editor",
                    "spreadsheet application",
                    "presentation software",
                    "email client interface",
                    "project management dashboard"
                ],
                "weight": 1.0
            },
            "Sports": {
                "descriptions": [
                    "sports news website",
                    "sports statistics dashboard",
                    "fantasy sports platform",
                    "sports betting interface",
                    "match schedule application",
                    "sports team page",
                    "athlete profile page",
                    "sports league website",
                    "game scoreboard",
                    "sports highlights platform",
                    "tournament bracket interface",
                    "sports analysis dashboard",
                    "fitness training tracker"
                ],
                "weight": 1.0
            },
            "Food & Dining": {
                "descriptions": [
                    "food delivery application",
                    "restaurant review website",
                    "recipe platform",
                    "restaurant reservation system",
                    "meal planning app",
                    "cooking tutorial interface",
                    "food blog",
                    "nutrition information page",
                    "restaurant menu",
                    "grocery shopping platform",
                    "food ordering system",
                    "culinary social network",
                    "dietary tracking application"
                ],
                "weight": 1.0
            },
            "Real Estate": {
                "descriptions": [
                    "property listing website",
                    "real estate agency portal",
                    "home search platform",
                    "apartment rental site",
                    "real estate market analysis",
                    "mortgage calculator tool",
                    "property management dashboard",
                    "home valuation tool",
                    "virtual property tour",
                    "floor plan viewer",
                    "real estate agent profile",
                    "property comparison tool",
                    "real estate investment platform"
                ],
                "weight": 1.0
            },
            "Automotive": {
                "descriptions": [
                    "car shopping website",
                    "vehicle listing platform",
                    "automotive parts store",
                    "car comparison tool",
                    "vehicle maintenance tracker",
                    "car rental service",
                    "automotive news site",
                    "vehicle specifications page",
                    "automobile review platform",
                    "car dealership website",
                    "vehicle history report",
                    "electric vehicle charging map",
                    "automotive forum"
                ],
                "weight": 1.0
            },
            "Gaming": {
                "descriptions": [
                    "video game interface",
                    "gaming platform storefront",
                    "game launcher",
                    "gaming community forum",
                    "game achievement screen",
                    "gaming leaderboard",
                    "game settings menu",
                    "game inventory screen",
                    "game map interface",
                    "game character selection",
                    "gaming news website",
                    "game modding platform",
                    "gaming strategy guide",
                    "game streaming interface"
                ],
                "weight": 1.0
            },
            "Government & Public Services": {
                "descriptions": [
                    "government website",
                    "public service portal",
                    "tax filing system",
                    "voting information site",
                    "permit application interface",
                    "public records database",
                    "municipal service platform",
                    "citizen feedback system",
                    "government agency page",
                    "public transportation planner",
                    "emergency services information",
                    "city planning dashboard",
                    "community resources directory"
                ],
                "weight": 1.0
            },
            "Art & Design": {
                "descriptions": [
                    "design software interface",
                    "digital art creation tool",
                    "graphic design platform",
                    "digital portfolio website",
                    "art gallery website",
                    "photo editing application",
                    "3D modeling software",
                    "illustration tool",
                    "design template library",
                    "creative project management",
                    "digital asset management",
                    "typography editor",
                    "color palette selector"
                ],
                "weight": 1.0
            }
        }
        
        # Try to load or save configuration
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.logger.info(f"Loaded configuration with {len(loaded_config)} domains")
                    
                    # Check for missing domains in loaded config and add them
                    updated = False
                    for domain, config in default_domains.items():
                        if domain not in loaded_config:
                            loaded_config[domain] = config
                            updated = True
                            self.logger.info(f"Added new domain: {domain}")
                    
                    # Make sure each domain has a weight
                    for domain in loaded_config:
                        if "weight" not in loaded_config[domain]:
                            loaded_config[domain]["weight"] = 1.0
                            updated = True
                    
                    if updated:
                        with open(self.config_path, 'w') as f:
                            json.dump(loaded_config, f, indent=4)
                            self.logger.info(f"Updated configuration file with new domains")
                    
                    return loaded_config
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_domains, f, indent=4)
                self.logger.info(f"Created new configuration with {len(default_domains)} domains")
        except Exception as e:
            self.logger.error(f"Could not save default config: {e}")
        
        return default_domains
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve categorization accuracy.
        """
        if not self.enable_preprocessing:
            return image
        
        try:
            # Convert PIL image to CV2 format
            img_array = np.array(image)
            if img_array.shape[2] == 4:  # If RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Apply preprocessing techniques
            # 1. Enhance contrast
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # 2. Sharpen the image
            kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced_img, -1, kernel)
            
            # Convert back to PIL
            processed_image = Image.fromarray(sharpened)
            return processed_image
        
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image  # Return original if preprocessing fails
    
    def categorize_screenshot(self, image_path: str) -> Dict[str, Union[str, float, List]]:
        """
        Categorize a screenshot using CLIP zero-shot classification with detailed probabilities.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with domain prediction, probability, and other details
        """
        try:
            # Prepare all possible descriptions
            all_descriptions = []
            domain_names = []
            
            for domain, config in self.domains.items():
                weight = config.get("weight", 1.0)
                for desc in config.get("descriptions", []):
                    all_descriptions.append(f"a screenshot related to {domain.lower()}: {desc}")
                    domain_names.append(domain)
            
            # Add a generic "other" category
            all_descriptions.append("a screenshot not related to any specific category")
            domain_names.append("Other")
            
            # Process image with potential preprocessing
            image = Image.open(image_path).convert("RGB")
            
            # Apply preprocessing if enabled
            processed_image = self.preprocess_image(image)
            
            # Process with CLIP
            inputs = self.processor(
                text=all_descriptions, 
                images=processed_image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to device
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Convert to numpy for easier manipulation
            probs_np = probs.cpu().numpy()
            
            # Aggregate probabilities by domain (since we have multiple descriptions per domain)
            domain_probs = {}
            for i, domain in enumerate(domain_names):
                if domain not in domain_probs:
                    domain_probs[domain] = 0
                domain_probs[domain] += probs_np[i]
            
            # Apply domain weights from config
            for domain in domain_probs:
                if domain != "Other" and domain in self.domains:
                    weight = self.domains[domain].get("weight", 1.0)
                    domain_probs[domain] *= weight
            
            # Sort domains by probability
            sorted_domains = sorted(domain_probs.items(), key=lambda x: x[1], reverse=True)
            top_domains = [d[0] for d in sorted_domains[:5]]
            top_probs = [float(d[1]) for d in sorted_domains[:5]]
            
            # Update statistics
            self.stats["total_processed"] += 1
            if top_probs[0] >= self.confidence_threshold:
                self.stats["successful_categorizations"] += 1
                if top_domains[0] not in self.stats["domain_counts"]:
                    self.stats["domain_counts"][top_domains[0]] = 0
                self.stats["domain_counts"][top_domains[0]] += 1
            else:
                self.stats["low_confidence_count"] += 1
            
            self.stats["confidence_scores"].append(top_probs[0])
            
            # Log results
            self.logger.info(f"Image: {os.path.basename(image_path)}")
            for domain, prob in zip(top_domains[:3], top_probs[:3]):
                self.logger.info(f"  {domain}: {prob:.2f}")
            
            return {
                "domain": top_domains[0],
                "probability": float(top_probs[0]),
                "top_domains": top_domains,
                "probabilities": top_probs,
                "threshold": self.confidence_threshold
            }
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {image_path}")
            return {"domain": "Error", "probability": 0.0, "top_domains": [], "probabilities": []}
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return {"domain": "Error", "probability": 0.0, "top_domains": [], "probabilities": []}
    
    def save_categorized_image(self, image_path: str, domain: str) -> str:
        """
        Save image to its corresponding domain folder.
        
        Args:
            image_path: Path to the image file
            domain: Domain category to save under
            
        Returns:
            Path to the saved image or empty string if failed
        """
        try:
            # Create domain-specific folder if it doesn't exist
            domain_folder = os.path.join(self.output_base_dir, domain)
            os.makedirs(domain_folder, exist_ok=True)
            
            # Generate unique filename
            filename = os.path.basename(image_path)
            unique_filename = filename
            counter = 1
            while os.path.exists(os.path.join(domain_folder, unique_filename)):
                name, ext = os.path.splitext(filename)
                unique_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            # Copy image to domain folder
            destination = os.path.join(domain_folder, unique_filename)
            shutil.copy2(image_path, destination)
            
            self.logger.info(f"Saved {filename} to {destination}")
            return destination
        
        except Exception as e:
            self.logger.error(f"Error saving {image_path}: {e}")
            return ""
    
    def process_images(self, image_paths: List[str]) -> Dict[str, Dict[str, Union[str, bool, float, List]]]:
        """
        Process multiple images, categorize and save them.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Dictionary mapping image paths to their categorization results
        """
        results = {}
        for path in image_paths:
            categorization = self.categorize_screenshot(path)
            
            # Only save if domain is not 'Error' and probability is above the threshold
            if categorization['domain'] != "Error" and categorization['probability'] >= self.confidence_threshold:
                saved_path = self.save_categorized_image(path, categorization['domain'])
                results[path] = {
                    "domain": categorization['domain'],
                    "probability": categorization['probability'],
                    "top_domains": categorization['top_domains'],
                    "probabilities": categorization['probabilities'],
                    "saved": bool(saved_path),
                    "saved_path": saved_path,
                    "threshold": self.confidence_threshold
                }
            else:
                results[path] = {
                    "domain": categorization['domain'],
                    "probability": categorization['probability'],
                    "top_domains": categorization['top_domains'],
                    "probabilities": categorization['probabilities'],
                    "saved": False,
                    "saved_path": "",
                    "threshold": self.confidence_threshold
                }
        
        # Generate analytics after processing
        if len(image_paths) > 0:
            self.generate_analytics()
        
        return results
    
    def provide_feedback(self, image_path: str, correct_domain: str, previous_domain: str = None) -> bool:
        """
        Add user feedback to improve categorization.
        
        Args:
            image_path: Path to the image file
            correct_domain: The correct domain according to user
            previous_domain: The previously predicted domain (if any)
            
        Returns:
            Boolean indicating success
        """
        if not self.enable_learning:
            return False
            
        try:
            # Ensure the correct domain exists in our config
            if correct_domain not in self.domains and correct_domain != "Other":
                self.logger.warning(f"Unknown domain in feedback: {correct_domain}")
                return False
            
            # Record feedback entry
            entry = {
                "image": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "correct_domain": correct_domain,
                "previous_domain": previous_domain
            }
            
            self.feedback_data["entries"].append(entry)
            
            # Update confusion matrix for analysis
            if previous_domain and previous_domain != correct_domain:
                confusion_key = f"{previous_domain}->{correct_domain}"
                if confusion_key not in self.feedback_data["corrections"]:
                    self.feedback_data["corrections"][confusion_key] = 0
                self.feedback_data["corrections"][confusion_key] += 1
            
            # Save feedback data
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=4)
            
            # Adjust domain weights based on feedback
            self._adjust_domain_weights(correct_domain, previous_domain)
            
            # Move the image to the correct domain if needed
            if previous_domain and previous_domain != correct_domain:
                # Check if the image has a saved path in the incorrect domain
                prev_domain_folder = os.path.join(self.output_base_dir, previous_domain)
                filename = os.path.basename(image_path)
                
                # Find the image in the previous domain folder
                prev_files = os.listdir(prev_domain_folder) if os.path.exists(prev_domain_folder) else []
                for file in prev_files:
                    if file.startswith(os.path.splitext(filename)[0]):
                        # Move to correct domain
                        src_path = os.path.join(prev_domain_folder, file)
                        self.save_categorized_image(src_path, correct_domain)
                        
                        # Optionally delete from incorrect domain
                        try:
                            os.remove(src_path)
                        except:
                            pass
                        
                        break
            
            self.logger.info(f"Feedback recorded: {os.path.basename(image_path)} → {correct_domain}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
            return False
    
    def _adjust_domain_weights(self, correct_domain: str, previous_domain: str = None) -> None:
        """
        Adjust domain weights based on feedback.
        
        Args:
            correct_domain: The correct domain
            previous_domain: The previously predicted domain (if incorrect)
        """
        if correct_domain not in self.domains:
            return
            
        # Increase weight for correct domain
        current_weight = self.domains[correct_domain].get("weight", 1.0)
        self.domains[correct_domain]["weight"] = min(current_weight * 1.05, 2.0)  # Cap at 2.0
        
        # Decrease weight for incorrect domain if provided
        if previous_domain and previous_domain != correct_domain and previous_domain in self.domains:
            current_weight = self.domains[previous_domain].get("weight", 1.0)
            self.domains[previous_domain]["weight"] = max(current_weight * 0.95, 0.5)  # Floor at 0.5
        
        # Save updated weights to config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.domains, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving updated weights: {e}")
    
    def generate_analytics(self) -> Dict[str, Any]:
        """
        Generate analytics data for model performance.
        
        Returns:
            Dictionary of analytics data
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analytics_file = os.path.join(self.analytics_dir, f"analytics_{timestamp}.json")
            
            # Calculate additional metrics
            accuracy = 0
            if self.stats["total_processed"] > 0:
                accuracy = self.stats["successful_categorizations"] / self.stats["total_processed"]
            
            avg_confidence = 0
            if len(self.stats["confidence_scores"]) > 0:
                avg_confidence = sum(self.stats["confidence_scores"]) / len(self.stats["confidence_scores"])
            
            # Prepare analytics data
            analytics = {
                "timestamp": timestamp,
                "total_processed": self.stats["total_processed"],
                "successful_categorizations": self.stats["successful_categorizations"],
                "low_confidence_count": self.stats["low_confidence_count"],
                "accuracy": accuracy,
                "average_confidence": avg_confidence,
                "domain_counts": self.stats["domain_counts"],
                "threshold": self.confidence_threshold,
                "model_name": self.model.__class__.__name__,
                "preprocessing_enabled": self.enable_preprocessing,
                "learning_enabled": self.enable_learning,
                "domain_weights": {d: c.get("weight", 1.0) for d, c in self.domains.items()}
            }
            
            # Save analytics
            with open(analytics_file, 'w') as f:
                json.dump(analytics, f, indent=4)
                
            self.logger.info(f"Analytics generated: {analytics_file}")
            
            # Generate confusion matrix visualization if enough feedback
            if len(self.feedback_data["entries"]) > 5:
                self._generate_confusion_matrix()
                
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error generating analytics: {e}")
            return {}
    
    def _generate_confusion_matrix(self) -> None:
        """
        Generate confusion matrix visualization from feedback data.
        """
        try:
            # Only generate if we have corrections
            if not self.feedback_data["corrections"]:
                return
                
            # Create matrix data
            all_domains = list(self.domains.keys()) + ["Other"]
            matrix = np.zeros((len(all_domains), len(all_domains)))
            
            domain_indices = {domain: i for i, domain in enumerate(all_domains)}
            
            # Fill in the confusion matrix
            for correction, count in self.feedback_data["corrections"].items():
                try:
                    pred_domain, true_domain = correction.split("->")
                    pred_idx = domain_indices.get(pred_domain, -1)
                    true_idx = domain_indices.get(true_domain, -1)
                    
                    if pred_idx >= 0 and true_idx >= 0:
                        matrix[pred_idx, true_idx] = count
                except:
                    continue
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            sns.heatmap(matrix, annot=True, fmt=".0f", 
                        xticklabels=all_domains, yticklabels=all_domains)
            
            plt.title("Confusion Matrix from User Feedback")
            plt.xlabel("Correct Domain")
            plt.ylabel("Predicted Domain")
            
            # Improve tick label readability
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            
            # Adjust layout and save
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.analytics_dir, f"confusion_matrix_{timestamp}.png"))
            plt.close()
            
            self.logger.info(f"Confusion matrix visualization generated")
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix: {e}")
    
    def adjust_threshold(self, new_threshold: float) -> bool:
        """
        Adjust the confidence threshold for categorization.
        
        Args:
            new_threshold: New threshold value (between 0 and 1)
            
        Returns:
            Boolean indicating success
        """
        if 0 <= new_threshold <= 1:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Threshold adjusted to {new_threshold}")
            return True
        else:
            self.logger.error(f"Invalid threshold value: {new_threshold}")
            return False
    
    def add_domain(self, domain_name: str, descriptions: List[str]) -> bool:
        """
        Add a new domain to the categorizer.
        
        Args:
            domain_name: Name of the new domain
            descriptions: List of text descriptions for the domain
            
        Returns:
            Boolean indicating success
        """
        if domain_name in self.domains:
            self.logger.warning(f"Domain {domain_name} already exists")
            return False
            
        # Add the new domain
        self.domains[domain_name] = {
            "descriptions": descriptions,
            "weight": 1.0
        }
        
        # Save the updated config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.domains, f, indent=4)
            self.logger.info(f"Added new domain: {domain_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving updated config: {e}")
            return False
    
    def update_domain(self, domain_name: str, descriptions: List[str] = None, weight: float = None) -> bool:
        """
        Update an existing domain with new descriptions or weight.
        
        Args:
            domain_name: Name of the domain to update
            descriptions: New list of text descriptions (optional)
            weight: New weight value (optional)
            
        Returns:
            Boolean indicating success
        """
        if domain_name not in self.domains:
            self.logger.warning(f"Domain {domain_name} does not exist")
            return False
            
        # Update domain fields
        if descriptions is not None:
            self.domains[domain_name]["descriptions"] = descriptions
            
        if weight is not None and 0.5 <= weight <= 2.0:
            self.domains[domain_name]["weight"] = weight
            
        # Save the updated config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.domains, f, indent=4)
            self.logger.info(f"Updated domain: {domain_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving updated config: {e}")
            return False
    
    def remove_domain(self, domain_name: str) -> bool:
        """
        Remove a domain from the categorizer.
        
        Args:
            domain_name: Name of the domain to remove
            
        Returns:
            Boolean indicating success
        """
        if domain_name not in self.domains:
            self.logger.warning(f"Domain {domain_name} does not exist")
            return False
            
        # Remove the domain
        del self.domains[domain_name]
        
        # Save the updated config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.domains, f, indent=4)
            self.logger.info(f"Removed domain: {domain_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving updated config: {e}")
            return False
    
    def batch_process_directory(self, directory_path: str) -> Dict[str, Dict]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            Dictionary of results for all processed images
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            self.logger.error(f"Invalid directory path: {directory_path}")
            return {}
            
        # Get all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
        image_paths = []
        
        for file in os.listdir(directory_path):
            full_path = os.path.join(directory_path, file)
            if os.path.isfile(full_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(full_path)
        
        self.logger.info(f"Found {len(image_paths)} images in {directory_path}")
        
        # Process all images
        return self.process_images(image_paths)
    
    def export_config(self, output_path: str = None) -> str:
        """
        Export the current domain configuration.
        
        Args:
            output_path: Optional path to save the exported config
            
        Returns:
            Path to the exported config file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.analytics_dir, f"config_export_{timestamp}.json")
            
        try:
            with open(output_path, 'w') as f:
                json.dump(self.domains, f, indent=4)
            self.logger.info(f"Configuration exported to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return ""
    
    def import_config(self, config_path: str) -> bool:
        """
        Import a domain configuration.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Boolean indicating success
        """
        if not os.path.exists(config_path):
            self.logger.error(f"Config file not found: {config_path}")
            return False
            
        try:
            with open(config_path, 'r') as f:
                new_config = json.load(f)
                
            # Validate the config structure
            for domain, config in new_config.items():
                if "descriptions" not in config or not isinstance(config["descriptions"], list):
                    self.logger.error(f"Invalid domain config for {domain}: missing descriptions")
                    return False
            
            # Update the domains
            self.domains = new_config
            
            # Save to the main config file
            with open(self.config_path, 'w') as f:
                json.dump(self.domains, f, indent=4)
                
            self.logger.info(f"Configuration imported from {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
        
        import json
import os
import shutil
from datetime import datetime

class ImageCategorizer:
    def __init__(self,):  # Keep your existing __init__ function
        self.feedback_file = "analytics/feedback.json"
        self.load_feedback_data()

    def load_feedback_data(self):
        """Load or create feedback.json"""
        if not os.path.exists(self.feedback_file):
            self.feedback_data = {"entries": [], "corrections": {}}
            with open(self.feedback_file, "w") as f:
                json.dump(self.feedback_data, f, indent=4)
        else:
            with open(self.feedback_file, "r") as f:
                self.feedback_data = json.load(f)

    def provide_feedback(self, image_path: str, correct_domain: str, previous_domain: str = None) -> bool:
        """
        Add user feedback to improve categorization.
        """
        try:
            # Ensure feedback file exists
            self.load_feedback_data()

            # Add feedback entry
            feedback_entry = {
                "image": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "correct_domain": correct_domain,
                "previous_domain": previous_domain
            }
            self.feedback_data["entries"].append(feedback_entry)

            # Track corrections
            if previous_domain and previous_domain != correct_domain:
                confusion_key = f"{previous_domain} -> {correct_domain}"
                self.feedback_data["corrections"][confusion_key] = (
                    self.feedback_data["corrections"].get(confusion_key, 0) + 1
                )

            # Save updated feedback data
            with open(self.feedback_file, "w") as f:
                json.dump(self.feedback_data, f, indent=4)

            # Move image to correct category folder
            correct_folder = os.path.join("categorized_images", correct_domain)
            os.makedirs(correct_folder, exist_ok=True)
            shutil.move(image_path, os.path.join(correct_folder, os.path.basename(image_path)))

            print(f"✅ Feedback recorded: {image_path} → {correct_domain}")
            return True
        except Exception as e:
            print(f"❌ Error recording feedback: {e}")
            return False


# Example usage
if __name__ == "__main__":
    categorizer = ImageCategorizer(
        confidence_threshold=0.15,
        enable_preprocessing=True,
        enable_learning=True
    )
    
    # Example: Process a single image
    # result = categorizer.categorize_screenshot("path/to/image.jpg")
    # print(f"Domain: {result['domain']} (Probability: {result['probability']:.2f})")
    
    # Example: Process a directory of images
    # results = categorizer.batch_process_directory("path/to/screenshots")
    # print(f"Processed {len(results)} images")
    
    # Example: Provide feedback for a misclassified image
    # categorizer.provide_feedback("path/to/image.jpg", "Technology", "Finance")
    
    # Example: Add a new domain
    # categorizer.add_domain("Weather", [
    #     "weather forecasting application",
    #     "meteorology dashboard",
    #     "climate data visualization",
    #     "weather map interface",
    #     "temperature tracking tool"
    # ])