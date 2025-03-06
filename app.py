import os
import argparse
import json
from categorizer import ImageCategorizer
from datetime import datetime
import glob

def main():
    """
    Main entry point for the Image Categorizer application.
    """
    parser = argparse.ArgumentParser(description="Image Categorizer - Organize screenshots by domain")
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--categorize", "-c", help="Path to image or directory to categorize")
    group.add_argument("--feedback", "-f", help="Provide feedback for an image")
    group.add_argument("--analyze", "-a", action="store_true", help="Generate analytics")
    group.add_argument("--config", action="store_true", help="Manage configuration")
    
    # Categorization options
    parser.add_argument("--threshold", "-t", type=float, default=0.15, 
                        help="Confidence threshold (0-1)")
    parser.add_argument("--no-preprocessing", action="store_true", 
                        help="Disable image preprocessing")
    parser.add_argument("--no-learning", action="store_true", 
                        help="Disable feedback-based learning")
    
    # Feedback options
    parser.add_argument("--correct-domain", "-d", help="Correct domain for feedback")
    parser.add_argument("--previous-domain", "-p", help="Previous incorrect domain")
    
    # Configuration options
    parser.add_argument("--add-domain", help="Add a new domain")
    parser.add_argument("--remove-domain", help="Remove a domain")
    parser.add_argument("--update-domain", help="Update a domain")
    parser.add_argument("--descriptions", help="Comma-separated domain descriptions")
    parser.add_argument("--weight", type=float, help="Domain weight (0.5-2.0)")
    parser.add_argument("--export-config", help="Export config to specified path")
    parser.add_argument("--import-config", help="Import config from specified path")
    parser.add_argument("--list-domains", action="store_true", help="List all domains")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize categorizer
    categorizer = ImageCategorizer(
        confidence_threshold=args.threshold,
        enable_preprocessing=not args.no_preprocessing,
        enable_learning=not args.no_learning
    )
    
    # Handle different operation modes
    if args.categorize:
        handle_categorize(categorizer, args.categorize)
    elif args.feedback:
        handle_feedback(categorizer, args)
    elif args.analyze:
        handle_analytics(categorizer)
    elif args.config:
        handle_config(categorizer, args)
    
def handle_categorize(categorizer, path):
    """
    Handle categorization of an image or directory.
    """
    if os.path.isdir(path):
        print(f"Processing directory: {path}")
        results = categorizer.batch_process_directory(path)
        print(f"Processed {len(results)} images")
        
        # Print summary
        if results:
            domains = {}
            for image_path, result in results.items():
                domain = result["domain"]
                if domain not in domains:
                    domains[domain] = 0
                domains[domain] += 1
            
            print("\nCategorization summary:")
            for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                print(f"  {domain}: {count} images")
    else:
        if not os.path.exists(path):
            print(f"Error: File {path} not found")
            return
            
        print(f"Processing image: {path}")
        result = categorizer.categorize_screenshot(path)
        
        print(f"\nDomain: {result['domain']} (Probability: {result['probability']:.2f})")
        print("Other possibilities:")
        for domain, prob in zip(result['top_domains'][1:4], result['probabilities'][1:4]):
            print(f"  {domain}: {prob:.2f}")
            
        if result['probability'] >= categorizer.confidence_threshold:
            saved_path = categorizer.save_categorized_image(path, result['domain'])
            if saved_path:
                print(f"\nImage saved to: {saved_path}")
        else:
            print(f"\nConfidence below threshold ({categorizer.confidence_threshold}), image not categorized")

def handle_feedback(categorizer, args):
    """
    Handle feedback for an image.
    """
    if not args.correct_domain:
        print("Error: --correct-domain is required for feedback")
        return
        
    if not os.path.exists(args.feedback):
        print(f"Error: Image {args.feedback} not found")
        return
        
    success = categorizer.provide_feedback(
        args.feedback, 
        args.correct_domain, 
        args.previous_domain
    )
    
    if success:
        print(f"Feedback recorded: {os.path.basename(args.feedback)} → {args.correct_domain}")
    else:
        print("Error recording feedback")

def handle_analytics(categorizer):
    """
    Generate and display analytics.
    """
    analytics = categorizer.generate_analytics()
    
    if not analytics:
        print("Error generating analytics")
        return
        
    print("\nAnalytics Summary:")
    print(f"Total processed: {analytics['total_processed']} images")
    print(f"Successful categorizations: {analytics['successful_categorizations']} images")
    print(f"Low confidence count: {analytics['low_confidence_count']} images")
    print(f"Accuracy: {analytics['accuracy']:.2f}")
    print(f"Average confidence: {analytics['average_confidence']:.2f}")
    
    print("\nDomain distribution:")
    for domain, count in sorted(analytics['domain_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count} images")
        
    print("\nDomain weights:")
    for domain, weight in sorted(analytics['domain_weights'].items()):
        print(f"  {domain}: {weight:.2f}")
        
    # Display most recent confusion matrix if available
    confusion_matrices = glob.glob(os.path.join(categorizer.analytics_dir, "confusion_matrix_*.png"))
    if confusion_matrices:
        latest_matrix = max(confusion_matrices, key=os.path.getctime)
        print(f"\nLatest confusion matrix: {latest_matrix}")

def handle_config(categorizer, args):
    """
    Handle configuration operations.
    """
    if args.list_domains:
        print("\nAvailable domains:")
        for domain, config in sorted(categorizer.domains.items()):
            desc_count = len(config.get("descriptions", []))
            weight = config.get("weight", 1.0)
            print(f"  {domain}: {desc_count} descriptions, weight={weight:.2f}")
    
    elif args.add_domain:
        if not args.descriptions:
            print("Error: --descriptions required when adding a domain")
            return
            
        descriptions = [d.strip() for d in args.descriptions.split(",")]
        success = categorizer.add_domain(args.add_domain, descriptions)
        
        if success:
            print(f"Domain added: {args.add_domain}")
        else:
            print(f"Error adding domain: {args.add_domain}")
    
    elif args.update_domain:
        descriptions = None
        if args.descriptions:
            descriptions = [d.strip() for d in args.descriptions.split(",")]
            
        success = categorizer.update_domain(
            args.update_domain, 
            descriptions=descriptions,
            weight=args.weight
        )
        
        if success:
            print(f"Domain updated: {args.update_domain}")
        else:
            print(f"Error updating domain: {args.update_domain}")
    
    elif args.remove_domain:
        success = categorizer.remove_domain(args.remove_domain)
        
        if success:
            print(f"Domain removed: {args.remove_domain}")
        else:
            print(f"Error removing domain: {args.remove_domain}")
    
    elif args.export_config:
        path = categorizer.export_config(args.export_config)
        
        if path:
            print(f"Configuration exported to: {path}")
        else:
            print("Error exporting configuration")
    
    elif args.import_config:
        success = categorizer.import_config(args.import_config)
        
        if success:
            print(f"Configuration imported from: {args.import_config}")
        else:
            print("Error importing configuration")
    
    else:
        print("No configuration operation specified. Use --list-domains, --add-domain, etc.")

if __name__ == "__main__":
    main()