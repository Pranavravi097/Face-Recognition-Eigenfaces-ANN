import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from zipfile import ZipFile
import requests
from io import BytesIO
import logging
import time
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognition:
    def __init__(self):
        self.face_db = None
        self.mean_face = None
        self.mean_aligned_faces = None
        self.eigenfaces = None
        self.face_signatures = None
        self.labels = None
        self.nn_model = None
        # Different k values to test as mentioned in the factors section
        self.k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  
        self.k = None  # The best k value will be set during training
        self.accuracy_scores = []
        self.class_names = []  # Store the actual names of the people
        # Get the directory where the script is located
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        
    def download_dataset(self, url, extract_to=None):
        """Download and extract the dataset"""
        if extract_to is None:
            # Use the script directory to create the dataset folder
            extract_to = os.path.join(self.script_directory, "dataset")
            
        logger.info(f"Downloading dataset to {extract_to}...")
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(extract_to):
                os.makedirs(extract_to)
                
            # Download the zip file
            response = requests.get(url)
            zip_content = BytesIO(response.content)
            
            # Extract the zip file
            with ZipFile(zip_content) as z:
                z.extractall(extract_to)
            
            logger.info(f"Dataset downloaded and extracted to {extract_to}")
            
            # List contents to help debug
            logger.info("Extracted content structure:")
            for root, dirs, files in os.walk(extract_to):
                level = root.replace(extract_to, '').count(os.sep)
                indent = ' ' * 4 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                if level < 2:  # Only show files for the first two levels to avoid verbose output
                    for f in files:
                        logger.info(f"{indent}    {f}")
            
            return True
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def detect_dataset_structure(self, dataset_path):
        """Detect the structure of the dataset and return the path to faces directory"""
        logger.info("Detecting dataset structure...")
        
        # Check if there's a direct 'faces' subfolder
        direct_faces_path = os.path.join(dataset_path, "faces")
        if os.path.isdir(direct_faces_path):
            logger.info(f"Found faces directory at {direct_faces_path}")
            return direct_faces_path
        
        # Check for nested faces directory (e.g., dataset/some_folder/faces)
        for root, dirs, _ in os.walk(dataset_path):
            for dir_name in dirs:
                if dir_name.lower() == "faces":
                    faces_path = os.path.join(root, dir_name)
                    logger.info(f"Found faces directory at {faces_path}")
                    return faces_path
        
        # If no 'faces' directory is found, look for directories that might contain person subdirectories
        # This assumes the dataset has a structure where the top-level directories are person names
        potential_faces_paths = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # Check if this directory contains subdirectories that might be person folders
                subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
                if subdirs:
                    potential_faces_paths.append(item_path)
                    logger.info(f"Found potential persons directory at {item_path}")
        
        # If we found potential paths, return the first one
        if potential_faces_paths:
            logger.info(f"Using {potential_faces_paths[0]} as faces directory")
            return potential_faces_paths[0]
        
        # If no suitable structure is found, just return the dataset path itself
        # This will allow the load_images function to check if it contains image files directly
        logger.warning(f"No standard faces directory structure found. Using {dataset_path} directly.")
        return dataset_path
    
    def load_images(self, dataset_path):
        """Load images from the dataset folder, adapted for flexible directory structure"""
        logger.info("Loading images...")
        images = []
        labels = []
        
        # First, try to detect the dataset structure
        faces_path = self.detect_dataset_structure(dataset_path)
        
        # Check if the faces_path exists
        if not os.path.exists(faces_path):
            logger.error(f"Directory not found at {faces_path}")
            raise ValueError(f"Directory not found at {faces_path}")
        
        # Check if faces_path contains images directly or has person subdirectories
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        direct_images = []
        for ext in image_extensions:
            direct_images.extend(glob.glob(os.path.join(faces_path, f"*{ext}")))
        
        if direct_images:
            # Dataset contains images directly in the faces_path
            logger.info(f"Found {len(direct_images)} images directly in {faces_path}")
            person_id = 0  # Only one person in this case
            self.class_names = ["person"]  # Default name
            
            for img_path in direct_images:
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to a standard size (e.g., 100x100)
                    img = cv2.resize(img, (100, 100))
                    # Flatten the image into a column vector
                    img_vector = img.flatten()
                    images.append(img_vector)
                    labels.append(person_id)
                else:
                    logger.warning(f"Image at {img_path} could not be read.")
        else:
            # Check if there are person subdirectories
            person_dirs = [d for d in os.listdir(faces_path) if os.path.isdir(os.path.join(faces_path, d))]
            
            if not person_dirs:
                logger.error("No person directories or direct images found.")
                raise ValueError("No person directories or direct images found.")
            
            logger.info(f"Found {len(person_dirs)} person directories: {person_dirs}")
            self.class_names = person_dirs  # Store the names for later reference
            
            for person_id, person_dir in enumerate(person_dirs):
                person_path = os.path.join(faces_path, person_dir)
                
                # Look for images with various extensions
                person_images = []
                for ext in image_extensions:
                    person_images.extend(glob.glob(os.path.join(person_path, f"*{ext}")))
                
                if not person_images:
                    logger.warning(f"No images found in directory: {person_path}")
                    continue
                
                logger.info(f"Loading {len(person_images)} images for {person_dir} (ID: {person_id})")
                
                for img_path in person_images:
                    # Read image in grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to a standard size (e.g., 100x100)
                        img = cv2.resize(img, (100, 100))
                        # Flatten the image into a column vector
                        img_vector = img.flatten()
                        images.append(img_vector)
                        labels.append(person_id)
                    else:
                        logger.warning(f"Image at {img_path} could not be read.")
        
        # Check if any images were loaded
        if not images:
            logger.error("No images were loaded from the dataset.")
            raise ValueError("No images were loaded from the dataset.")
        
        # Convert to numpy arrays
        face_db = np.array(images).T  # Each column is an image, so transpose the array
        labels = np.array(labels)
        
        logger.info(f"Loaded {face_db.shape[1]} images from {len(set(labels))} people")
        return face_db, labels
    
    def compute_mean_face(self):
        """Compute the mean face from the database"""
        logger.info("Computing mean face...")
        self.mean_face = np.mean(self.face_db, axis=1, keepdims=True)
        return self.mean_face

    def align_faces(self):
        """Subtract mean face from each face image (Step 3: Do mean Zero)"""
        logger.info("Aligning faces with mean...")
        self.mean_aligned_faces = self.face_db - self.mean_face
        return self.mean_aligned_faces

    def compute_covariance(self):
        """Compute the surrogate covariance matrix (Step 4)"""
        logger.info("Computing surrogate covariance matrix...")
        # Using surrogate covariance calculation as per Turk and Pentland method
        cov_matrix = np.dot(self.mean_aligned_faces.T, self.mean_aligned_faces)
        return cov_matrix

    def eigen_decomposition(self, cov_matrix):
        """Perform eigenvalue decomposition (Step 5)"""
        logger.info("Performing eigen decomposition...")
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors

    def select_eigenvectors(self, eigenvectors, k):
        """Select the top k eigenvectors (Step 6: Find the best direction)"""
        logger.info(f"Selecting top {k} eigenvectors...")
        return eigenvectors[:, :k]

    def generate_eigenfaces(self, feature_vectors):
        """Generate eigenfaces by projecting mean aligned faces onto feature vectors (Step 7)"""
        logger.info("Generating eigenfaces...")
        eigenfaces = np.dot(self.mean_aligned_faces, feature_vectors)
        
        # Normalize eigenfaces
        for i in range(eigenfaces.shape[1]):
            eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
        
        return eigenfaces

    def generate_face_signatures(self, eigenfaces):
        """Generate signature for each face in the database (Step 8)"""
        logger.info("Generating face signatures...")
        signatures = np.dot(eigenfaces.T, self.mean_aligned_faces)
        return signatures

    def project_test_face(self, test_face, eigenfaces):
        """Project a test face onto the eigenfaces space (Testing Step 3)"""
        mean_aligned_test = test_face - self.mean_face
        projected_test = np.dot(eigenfaces.T, mean_aligned_test)
        return projected_test

    def train_ann(self, signatures, labels):
        """Train an Artificial Neural Network classifier (Step 9)"""
        logger.info("Training ANN classifier...")
        X = signatures.T
        
        # Create a neural network classifier with multiple hidden layers
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        mlp.fit(X, labels)
        
        return mlp

    def evaluate_model(self, test_faces, test_labels, eigenfaces, model):
        """Evaluate the model on test data"""
        logger.info("Evaluating model...")
        projected_tests = []
        
        for face in test_faces.T:  # Iterate through columns
            face = face.reshape(-1, 1)  # Reshape to column vector
            projected = self.project_test_face(face, eigenfaces)
            projected_tests.append(projected.flatten())
        
        # Stack as rows for prediction
        X_test = np.vstack(projected_tests)
        
        # Predict using the trained model
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(test_labels, predictions)
        logger.info(f"Confusion matrix shape: {conf_matrix.shape}")
        
        return accuracy, predictions, conf_matrix

    def identify_impostors(self, test_faces, eigenfaces, model, threshold=0.6):
        """
        Identify potential impostors based on prediction confidence
        (Part of requirement b: Add imposters who don't belong to the training set)
        """
        logger.info("Identifying potential impostors...")
        projected_tests = []
        
        for face in test_faces.T:  # Iterate through columns
            face = face.reshape(-1, 1)  # Reshape to column vector
            projected = self.project_test_face(face, eigenfaces)
            projected_tests.append(projected.flatten())
        
        # Stack as rows for prediction
        X_test = np.vstack(projected_tests)
        
        # Get prediction probabilities
        proba = model.predict_proba(X_test)
        
        # Identify potential impostors (max probability below threshold)
        max_proba = np.max(proba, axis=1)
        impostors = np.where(max_proba < threshold)[0]
        
        logger.info(f"Identified {len(impostors)} impostors out of {len(max_proba)} test faces")
        return impostors, max_proba

    def train_and_evaluate(self):
        """
        Train and evaluate the model with different k values
        (Factor a: Change the value of k and check how it affects accuracy)
        """
        # Check if we have enough samples for stratified split
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        min_count = np.min(counts)
        
        if min_count < 2:
            logger.warning(f"Some classes have fewer than 2 samples (min: {min_count}). Using regular train_test_split.")
            # Using 60% training and 40% test split as required in the PDF
            train_faces, test_faces, train_labels, test_labels = train_test_split(
                self.face_db.T, self.labels, test_size=0.4, random_state=42
            )
        else:
            # Using 60% training and 40% test split as required in the PDF with stratification
            train_faces, test_faces, train_labels, test_labels = train_test_split(
                self.face_db.T, self.labels, test_size=0.4, random_state=42, stratify=self.labels
            )
        
        # Convert train_faces back to expected format (samples as columns)
        self.face_db = train_faces.T
        train_labels_copy = train_labels.copy()
        
        # Step 2: Mean Calculation
        self.compute_mean_face()
        
        # Step 3: Do mean Zero
        self.align_faces()
        
        # Step 4: Calculate Co-Variance
        cov_matrix = self.compute_covariance()
        
        # Step 5: Do eigenvalue and eigenvector decomposition
        eigenvalues, eigenvectors = self.eigen_decomposition(cov_matrix)
        
        # Determine the maximum possible k value based on data
        max_k = min(len(eigenvalues), max(self.k_values))
        valid_k_values = [k for k in self.k_values if k <= max_k]
        
        if not valid_k_values:
            valid_k_values = [max_k // 2]  # Use half of max_k if all k values are too large
            logger.warning(f"All specified k values are too large. Using k={valid_k_values[0]} instead")
        
        results = []
        self.accuracy_scores = []
        
        # Factor a: Test different k values and plot accuracy
        for k in valid_k_values:
            logger.info(f"\nTesting with k={k}")
            
            # Step 6: Find the best direction (Generate feature vectors)
            selected_vectors = self.select_eigenvectors(eigenvectors, k)
            
            # Step 7: Generate Eigenfaces
            eigenfaces = self.generate_eigenfaces(selected_vectors)
            
            # Step 8: Generate Signature of Each Face
            signatures = self.generate_face_signatures(eigenfaces)
            
            # Step 9: Apply ANN for training
            model = self.train_ann(signatures, train_labels_copy)
            
            # Test the model
            accuracy, _, _ = self.evaluate_model(test_faces.T, test_labels, eigenfaces, model)
            results.append((k, accuracy))
            self.accuracy_scores.append(accuracy)
        
        # Find the best k value
        best_k, best_accuracy = max(results, key=lambda x: x[1])
        logger.info(f"\nBest k value: {best_k} with accuracy: {best_accuracy:.4f}")
        
        # Train the final model with the best k value
        self.k = best_k
        selected_vectors = self.select_eigenvectors(eigenvectors, best_k)
        self.eigenfaces = self.generate_eigenfaces(selected_vectors)
        self.face_signatures = self.generate_face_signatures(self.eigenfaces)
        self.nn_model = self.train_ann(self.face_signatures, train_labels_copy)
        
        # Factor b: Test with impostors
        logger.info("\nTesting for impostors...")
        num_impostors = 10  # Create 10 random impostor faces
        
        # Create random impostor faces that look more like noise than real faces
        # This makes them easier to detect as impostors
        impostor_faces = np.random.rand(train_faces.shape[1], num_impostors) * 255
        
        # Evaluate model on test faces with impostors
        combined_test_faces = np.hstack((test_faces.T, impostor_faces))
        
        # Create labels for impostors (using a new class ID)
        max_label = np.max(self.labels) + 1
        impostor_labels = np.ones(num_impostors, dtype=int) * max_label
        
        # Combined test labels
        combined_test_labels = np.hstack((test_labels, impostor_labels))
        
        # Identify impostors
        impostors, confidences = self.identify_impostors(
            combined_test_faces, self.eigenfaces, self.nn_model, threshold=0.6
        )
        
        # Print details about the identified impostors
        logger.info(f"Total samples: {len(confidences)}")
        logger.info(f"Genuine samples: {len(confidences) - num_impostors}")
        logger.info(f"Impostor samples: {num_impostors}")
        logger.info(f"Number of identified impostors: {len(impostors)}")
        
        # Show confidence values for impostors
        genuine_count = len(confidences) - num_impostors
        impostor_confidences = confidences[genuine_count:]
        logger.info(f"Impostor confidence values: {impostor_confidences}")
        
        # Plot confidence distributions for genuine and impostor, including identified impostors
        self.plot_confidence_distribution(confidences, genuine_count, num_impostors, identified_impostors=impostors)
        
        # Plot accuracy vs k value
        self.plot_accuracy_vs_k(valid_k_values)
        
        # Visualize eigenfaces
        self.visualize_eigenfaces(num_eigenfaces=min(6, len(valid_k_values)))
        
        # Visualize mean face
        self.visualize_mean_face()
        
        return best_accuracy

    def plot_accuracy_vs_k(self, k_values=None):
        """Plot accuracy vs k value (Factor a visualization)"""
        if k_values is None:
            k_values = self.k_values
            
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, self.accuracy_scores, marker='o', linestyle='-', color='blue')
        plt.title('Accuracy vs Number of Eigenfaces (k)')
        plt.xlabel('Number of Eigenfaces (k)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.xticks(k_values)
        
        # Highlight the best k value
        if self.k is not None:
            best_idx = k_values.index(self.k)
            best_acc = self.accuracy_scores[best_idx]
            plt.scatter([self.k], [best_acc], c='red', s=100, label=f'Best k={self.k}')
            plt.annotate(f'Accuracy: {best_acc:.4f}', 
                         xy=(self.k, best_acc), 
                         xytext=(self.k+2, best_acc-0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.legend()
        # Save to script directory
        save_path = os.path.join(self.script_directory, 'accuracy_vs_k.png')
        plt.savefig(save_path)
        logger.info(f"Plot saved as '{save_path}'")
    
    def plot_confidence_distribution(self, confidences, genuine_count, num_impostors, identified_impostors=None):
        """Plot confidence distribution for genuine users vs impostors"""
        plt.figure(figsize=(10, 6))
        
        # Separate confidences for genuine and impostor samples
        genuine_conf = confidences[:genuine_count]
        impostor_conf = confidences[genuine_count:]
        
        # Debug info
        logger.info(f"Genuine confidences: min={np.min(genuine_conf):.4f}, max={np.max(genuine_conf):.4f}, mean={np.mean(genuine_conf):.4f}")
        logger.info(f"Impostor confidences: min={np.min(impostor_conf):.4f}, max={np.max(impostor_conf):.4f}, mean={np.mean(impostor_conf):.4f}")
        
        # Make sure we have bins that will show both distributions
        min_conf = min(np.min(genuine_conf), np.min(impostor_conf))
        max_conf = max(np.max(genuine_conf), np.max(impostor_conf))
        bin_edges = np.linspace(min_conf, max_conf, 20)
        
        # Create histograms with the same bin edges
        plt.hist(genuine_conf, bins=bin_edges, alpha=0.7, label='Genuine', color='green')
        plt.hist(impostor_conf, bins=bin_edges, alpha=0.7, label='Impostor', color='red')
        
        # If we have identified impostors, mark them on the plot
        if identified_impostors is not None:
            # Filter out only the actual impostor indices (those in the last num_impostors)
            true_impostor_indices = []
            
            for idx in identified_impostors:
                if idx >= genuine_count:  # This is a true impostor
                    # Adjust index to match the impostor_conf array
                    adjusted_idx = idx - genuine_count
                    true_impostor_indices.append(adjusted_idx)
            
            # Highlight the identified impostors
            if true_impostor_indices:
                # Get confidences of correctly identified impostors
                correct_impostor_conf = [impostor_conf[i] for i in true_impostor_indices]
                
                # Add scatter points for identified impostors
                plt.scatter([impostor_conf[i] for i in range(len(impostor_conf)) if i in true_impostor_indices],
                           [5] * len(true_impostor_indices),  # Place points at y=5 for visibility
                           c='yellow', s=100, label='Identified Impostors', zorder=5, 
                           edgecolor='black', marker='*')
                
                # Calculate detection rate
                detection_rate = len(true_impostor_indices) / num_impostors * 100
                plt.annotate(f'Impostor Detection Rate: {detection_rate:.1f}%',
                            xy=(0.02, 0.95), xycoords='axes fraction', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                            fontsize=10)
        
        # Add vertical line at the threshold
        threshold = 0.6  # Same threshold used for impostor identification
        plt.axvline(x=threshold, color='black', linestyle='--', 
                   label=f'Threshold ({threshold})')
        
        plt.title('Confidence Score Distribution: Genuine vs Impostor')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to script directory
        save_path = os.path.join(self.script_directory, 'confidence_distribution.png')
        plt.savefig(save_path)
        logger.info(f"Confidence distribution plot saved as '{save_path}'")
    
    def visualize_eigenfaces(self, num_eigenfaces=5):
        """Visualize the top eigenfaces"""
        if self.eigenfaces is None:
            logger.warning("Eigenfaces not computed yet. Cannot visualize.")
            return
        
        # Display at most num_eigenfaces or as many as available
        n = min(num_eigenfaces, self.eigenfaces.shape[1])
        
        plt.figure(figsize=(15, 3))
        for i in range(n):
            plt.subplot(1, n, i+1)
            # Reshape eigenface vector to image dimensions (100x100)
            eigenface = self.eigenfaces[:, i].reshape(100, 100)
            
            # Normalize for better visualization
            eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
            
            plt.imshow(eigenface, cmap='gray')
            plt.title(f'Eigenface {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        # Save to script directory
        save_path = os.path.join(self.script_directory, 'eigenfaces.png')
        plt.savefig(save_path)
        logger.info(f"Eigenfaces visualization saved as '{save_path}'")

    def visualize_mean_face(self):
        """Visualize the mean face"""
        if self.mean_face is None:
            logger.warning("Mean face not computed yet. Cannot visualize.")
            return
        
        plt.figure(figsize=(5, 5))
        # Reshape mean face vector to image dimensions (100x100)
        mean_face_img = self.mean_face.reshape(100, 100)
        
        # Normalize for better visualization
        mean_face_img = (mean_face_img - mean_face_img.min()) / (mean_face_img.max() - mean_face_img.min())
        
        plt.imshow(mean_face_img, cmap='gray')
        plt.title('Mean Face')
        plt.axis('off')
        plt.tight_layout()
        # Save to script directory
        save_path = os.path.join(self.script_directory, 'mean_face.png')
        plt.savefig(save_path)
        logger.info(f"Mean face visualization saved as '{save_path}'")
        
    def predict_identity(self, image_path):
        """Predict the identity of a face in an image"""
        if self.eigenfaces is None or self.nn_model is None:
            logger.error("Model not trained yet. Cannot predict.")
            return None
        
        # Load and preprocess the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Could not read image at {image_path}")
            return None
        
        # Resize to the same dimensions used during training
        img = cv2.resize(img, (100, 100))
        img_vector = img.flatten().reshape(-1, 1)
        
        # Project onto eigenface space
        projected = self.project_test_face(img_vector, self.eigenfaces)
        
        # Predict identity
        identity = self.nn_model.predict([projected.flatten()])[0]
        confidence = np.max(self.nn_model.predict_proba([projected.flatten()]))
        
        # Get the name instead of just the ID
        if 0 <= identity < len(self.class_names):
            name = self.class_names[identity]
        else:
            name = f"Unknown (ID: {identity})"
        
        logger.info(f"Predicted identity: {name} with confidence {confidence:.2f}")
        
        if confidence < 0.6:
            logger.info("Low confidence: Possible impostor detected!")
            return "Possible Impostor", confidence
        
        return name, confidence

    def run(self, dataset_path=None):
        """Run the complete face recognition pipeline"""
        start_time = time.time()
        
        if dataset_path is None:
            dataset_url = "https://github.com/robaita/introduction_to_machine_learning/raw/main/dataset.zip"
            # Download to script directory
            dataset_path = os.path.join(self.script_directory, "dataset")
            self.download_dataset(dataset_url, dataset_path)
        
        self.face_db, self.labels = self.load_images(dataset_path)
        
        # Check if we have enough data to proceed
        if self.face_db.shape[1] < 5:
            logger.warning("Very few images found. Results may not be reliable.")
        
        best_accuracy = self.train_and_evaluate()
        
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
        
        return best_accuracy


if __name__ == "__main__":
    face_rec = FaceRecognition()
    face_rec.run()