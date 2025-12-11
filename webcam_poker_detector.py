import cv2 
from ultralytics import YOLO 
from poker_hands import classify_hand
from typing import List, Tuple
import time 



class WebcamPokerDetector:
    #Real time poker hand detection using webcam

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt", 
                 conf_threshold: float = 0.5, 
                 camera_id: int = 0):
        

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold 
        self.camera_id = camera_id

        self.fps = 0 
        self.prev_time = 0

         # ========== NEW CODE: Load special image ==========
        special_image_path = "images.jpeg"  # ← CHANGE THIS to your image path!
        self.special_image = cv2.imread(special_image_path)
        
        if self.special_image is None:
            print(f"Warning: Could not load special image at {special_image_path}")
            print("Please update the path in __init__ method")
        else:
            # Resize to a good size for popup (optional)
            self.special_image = cv2.resize(self.special_image, (800, 600))
            print(f"✓ Loaded special image: {special_image_path}")
    # ==================================================

    
    def _parse_card_name(self, card_name: str) -> Tuple[str, str]:
        """Parse card name into (rank, suit)."""
        if card_name.startswith('10'):
            rank = '10'
            suit = card_name[2]
        else: 
            rank = card_name[:-1]
            suit = card_name[-1]
        return rank, suit
    
    def _detect_cards_from_frame(self, frame):
         
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False
        )
        
        cards = []
        result = results[0]
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if confidence >= self.conf_threshold:
                card_name = result.names[cls_id]
                rank, suit = self._parse_card_name(card_name)
                cards.append((rank, suit))
        
        return cards, result  # Returns tuple of (list, result)
        
    
    def _draw_info(self, frame, cards, hand_type, fps):
        
        height, width = frame.shape[:2]

        #Create info box background 
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

       # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display number of cards detected
        color = (0, 255, 0) if len(cards) == 5 else (0, 165, 255)
        cv2.putText(frame, f"Cards Detected: {len(cards)}/5", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display cards
        if cards:
            cards_str = ", ".join([f"{rank}{suit}" for rank, suit in cards])
            cv2.putText(frame, f"Cards: {cards_str}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display poker hand (if 5 cards)
        if len(cards) == 5 and hand_type:
            hand_color = (0, 255, 255)  # Yellow for valid hand
            cv2.putText(frame, f"Hand: {hand_type}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
            

         # ========== NEW CODE: Check for 2 and show popup ==========
        has_two = any(rank == '2' for rank, suit in cards)
        
        if has_two and self.special_image is not None:
            # Create a copy of the image for display
            display_img = self.special_image.copy()
            
            # Add text overlay on the popup image
            img_height, img_width = display_img.shape[:2]
            cv2.putText(display_img, "2 DETECTED!", 
                        (img_width//2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            # Show in separate window
            cv2.imshow("2 DETECTED - Special Image!", display_img)
        else:
            # Close the popup window if no 2 detected
            try:
                cv2.destroyWindow("2 DETECTED - Special Image!")
            except:
                pass
    # ==========================================================   
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save screenshot", 
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    
    def run(self):
        """
        Start live webcam detection.
        
        Controls:
        - 'q': Quit
        - 's': Save screenshot
        - '+': Increase confidence threshold
        - '-': Decrease confidence threshold
        """

        #Open webcam 
        cap = cv2.VideoCapture(self.camera_id)


        if not cap.isOpened():
            print(f"Error: Could not open Camera {self.camera_id}")
            return 
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # Height
        
        print("=" * 60)
        print("Webcam Poker Detector Started!")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '+' - Increase confidence threshold")
        print("  '-' - Decrease confidence threshold")
        print("=" * 60)

        screenshot_count = 0

        try:
            while True: 
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                  # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
                self.prev_time = current_time
                
                # Detect cards
                cards, result = self._detect_cards_from_frame(frame)
                
                # Classify poker hand if 5 cards detected
                hand_type = None
                if len(cards) == 5:
                    hand_type = classify_hand(cards)
                
                # Draw YOLO detections (bounding boxes)
                annotated_frame = result.plot()
                
                # Draw info overlay
                self._draw_info(annotated_frame, cards, hand_type, fps)
                
                # Display frame
                cv2.imshow("Poker Hand Detector", annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_count += 1
                    filename = f"poker_hand_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Screenshot saved: {filename}")
                    
                    # Also print current detection
                    if len(cards) == 5:
                        print(f"  Cards: {cards}")
                        print(f"  Hand: {hand_type}")
                elif key == ord('+') or key == ord('='):
                    # Increase confidence threshold
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease confidence threshold
                    self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released. Goodbye!")


class VideoFilePokerDetector(WebcamPokerDetector):
    """
    Poker hand detection from video file (extends WebcamPokerDetector).
    """
    
    def __init__(self, model_path: str, video_path: str, conf_threshold: float = 0.5):
        """
        Initialize video file poker detector.
        
        Args:
            model_path: Path to trained YOLO model
            video_path: Path to video file
            conf_threshold: Confidence threshold for detections
        """
        super().__init__(model_path, conf_threshold)
        self.video_path = video_path
    
    def run(self):
        """Process video file instead of webcam."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        print(f"Processing video: {self.video_path}")
        print("Press 'q' to quit, 's' to save screenshot")
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 2 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time
            
            # Detect cards
            cards, result = self._detect_cards_from_frame(frame)
            
            # Classify poker hand
            hand_type = None
            if len(cards) == 5:
                hand_type = classify_hand(cards)
            
            # Draw detections and info
            annotated_frame = result.plot()
            self._draw_info(annotated_frame, cards, hand_type, fps)
            
            # Display
            cv2.imshow("Poker Hand Detector - Video", annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"video_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("POKER HAND DETECTOR - LIVE MODE")
    print("=" * 60)
    
    # Check if model exists
    model_path = "runs/detect/train/weights/best.pt"
    
    print(f"\nLooking for model at: {model_path}")
    
    try:
        # Choose mode
        if len(sys.argv) > 1:
            if sys.argv[1] == "video" and len(sys.argv) > 2:
                # Video file mode
                video_path = sys.argv[2]
                detector = VideoFilePokerDetector(
                    model_path=model_path,
                    video_path=video_path,
                    conf_threshold=0.5
                )
            else:
                print("Usage: python webcam_poker_detector.py [video <path>]")
                sys.exit(1)
        else:
            # Webcam mode (default)
            detector = WebcamPokerDetector(
                model_path=model_path,
                conf_threshold=0.5,
                camera_id=0
            )
        
        # Run detector
        detector.run()
    
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        print("Please train your model first using train_model.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()