"""
Invoice extraction module using YOLO and Vision Language Model.
Cross-platform optimized for Mac and Windows.
"""

import time
import json
import re
import os
import shutil
import torch
from ultralytics import YOLO
from transformers import Qwen2_5_VLForConditionalGeneration , AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.preprocessing import preprocess_pipeline
from utils.utilities import get_optimal_device


class InvoiceExtractor:
    """Extract structured information from tractor invoice images."""
    
    def __init__(self, yolo_path, vlm_model_name="Qwen/Qwen2.5-VL-3B-Instruct", device="auto"):
        """
        Initialize the invoice extractor.
        
        Args:
            yolo_path: Path to YOLO model weights
            vlm_model_name: Hugging Face model name for VLM
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        """
        print("⏳ Loading Models...")
        
        # Determine optimal device
        self.device = get_optimal_device(device)
        print(f"📍 Using device: {self.device}")
        
        # Load YOLO model
        print("   Loading YOLO model...")
        self.yolo = YOLO(yolo_path)
        
        # Move YOLO to appropriate device
        if self.device == "cuda":
            self.yolo.to('cuda')
        
        # Load VLM model with cross-platform optimization
        print("   Loading Vision Language Model...")
        self.vlm = self._load_vlm_model(vlm_model_name)
        self.processor = AutoProcessor.from_pretrained(
            vlm_model_name, 
            trust_remote_code=True
        )
        
        print("✅ Models Loaded Successfully")
        print()
    
    def _load_vlm_model(self, model_name):
        """Load VLM model with device-specific optimizations."""
        if self.device == "cuda":
            # Windows/Linux with NVIDIA GPU - Use 4-bit quantization
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto",      # IMPORTANT
                    low_cpu_mem_usage=True,
                )
                  
                return model
                
            except Exception as e:
                print(f"   ⚠️  Quantization failed: {e}")
                print("   → Falling back to float16")
                return self._load_fallback_cuda(model_name)
        
        elif self.device == "mps":
            # Mac with Apple Silicon - Critical fixes for size mismatch
            print("   → Loading for Apple Silicon (MPS)")
            print("   → Expected memory: ~6-8 GB unified memory")
            
            try:
                # CRITICAL: Use float32 for MPS to avoid precision issues
                # Load on CPU first to avoid MPS allocation problems
                print("   → Step 1: Loading model to CPU...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use float32 for MPS stability
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=None,  # Critical: Load to CPU first
                    local_files_only=False,  # Allow downloading if needed
                )
                
                # Move to MPS after loading completes
                print("   → Step 2: Moving model to MPS device...")
                model = model.to("mps")
                print("   ✓ Model successfully loaded on MPS")
                return model
                
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"   ❌ Size mismatch error detected!")
                    print(f"   → Error: {e}")
                    print()
                    print("   🔧 Attempting to fix by forcing fresh download...")
                    print("   → This will bypass the corrupted cache")
                    print()
                    try:
                        # Force fresh download to bypass corrupted cache
                        # Clear the specific model cache
                        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
                        
                        if os.path.exists(model_cache):
                            print(f"   → Removing corrupted cache: {model_cache}")
                            try:
                                shutil.rmtree(model_cache)
                                print("   ✓ Cache cleared successfully")
                            except PermissionError:
                                print("   ⚠️  Could not delete cache (permission denied)")
                                print("   → Will force re-download anyway")
                        
                        # Force fresh download
                        print("   → Downloading fresh model files...")
                        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map=None,
                            force_download=True,  # Force fresh download
                            resume_download=False,
                        )
                        
                        # Move to MPS after loading
                        print("   → Moving model to MPS device...")
                        model = model.to("mps")
                        print("   ✓ Model successfully loaded on MPS (fresh download)")
                        return model
                        
                    except Exception as retry_error:
                        print(f"   ❌ Retry with fresh download failed: {retry_error}")
                        print()
                        print("   🔧 MANUAL SOLUTION: Clear your Hugging Face cache:")
                        print("   → Run: rm -rf ~/.cache/huggingface/hub/models--Qwen--*")
                        print("   → Or use: huggingface-cli delete-cache")
                        raise RuntimeError(
                            "Model cache is corrupted and automatic fix failed.\n"
                            "Please manually clear cache: rm -rf ~/.cache/huggingface/hub/models--Qwen--*"
                        )
                else:
                    print(f"   ⚠️  MPS loading failed: {e}")
                    print("   → Trying CPU fallback...")
                    return self._load_fallback_cpu(model_name)
            
            except Exception as e:
                print(f"   ⚠️  Unexpected error: {e}")
                print("   → Trying CPU fallback...")
                return self._load_fallback_cpu(model_name)
        
        else:
            # CPU fallback
            print("   → Using CPU with float32 precision")
            print("   → Expected memory: ~12-14 GB RAM")
            print("   ⚠️  Warning: CPU inference will be slow")
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            return model
    
    def _load_fallback_cuda(self, model_name):
        """Fallback method for CUDA without quantization."""
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        return model
    
    def _load_fallback_cpu(self, model_name):
        """Fallback method for CPU loading."""
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        return model
    
    def _load_fallback_mps(self, model_name):
        """Fallback method for MPS with float32."""
        try:
            # Try float32 as last resort for MPS
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None,
            )
            model = model.to("mps")
            return model
        except Exception as e:
            print(f"   ❌ All MPS loading methods failed: {e}")
            print("   → Loading to CPU instead")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="cpu",
            )
            return model
    
    def _clean_number(self, value):
        """
        Convert strings like '50 HP', 'Rs. 5,25,000/-' to pure numbers.
        
        Args:
            value: String or numeric value to clean
            
        Returns:
            Cleaned numeric value (int or float) or None
        """
        if not value:
            return None
        try:
            # Remove all non-digit and non-decimal characters
            clean_str = re.sub(r'[^\d.]', '', str(value))
            if not clean_str:
                return None
            if "." in clean_str:
                return float(clean_str)
            return int(clean_str)
        except (ValueError, AttributeError):
            return None
    
    def run(self, raw_image_path, enable_preprocessing=False):
        """
        Run full extraction pipeline on an invoice image.
        
        Args:
            raw_image_path: Path to input image
            enable_preprocessing: Whether to apply image preprocessing
            
        Returns:
            Dictionary containing extracted fields and metadata
        """
        start_time = time.time()
        
        # Initialize output structure
        final_json = {
            "doc_id": raw_image_path.split("/")[-1].split(".")[0],
            "fields": {
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None,
                "signature": {"present": False, "bbox": []},
                "stamp": {"present": False, "bbox": []}
            },
            "confidence": 0.00,
            "processing_time_sec": 0,
            "cost_estimate_usd": 0
        }
        
        # Preprocessing (optional)
        if enable_preprocessing:
            processed_path = preprocess_pipeline(raw_image_path)
        else:
            processed_path = raw_image_path
        
        # YOLO Detection for Signatures/Stamps
        yolo_results = self.yolo(processed_path, verbose=False)[0]
        max_yolo_conf = 0.0
        
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            label = yolo_results.names[cls_id].lower()
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            bbox = [int(x) for x in bbox]
            
            if conf > 0.4:  # Detection threshold
                if label in final_json["fields"]:
                    final_json["fields"][label] = {
                        "present": True,
                        "bbox": bbox
                    }
                    max_yolo_conf = max(max_yolo_conf, conf)
        
        # VLM Extraction for Text Fields
        prompt_text = """ 
ROLE:
You are a multilingual tractor invoice OCR extraction system.

SUPPORTED SCRIPTS:
Latin, Devanagari (हिन्दी), Gujarati (ગુજરાતી), Odia (ଓଡ଼ିଆ), Gurmukhi (ਪੰਜਾਬੀ), and other Indian scripts.

LANGUAGE PRESERVATION & PRIORITY (DO NOT TRANSLATE):

If a field appears in multiple languages/scripts in the image:
• ALWAYS prefer the NATIVE / REGIONAL script over English.
• Hindi (Devanagari) has priority if present.
• Preserve the script exactly as seen in the image.

EXAMPLES:
• Hindi: "भारत ट्रैक्टर्स" → "भारत ट्रैक्टर्स"
• Gujarati: "શ્રી ગણેશ ટ્રેક્ટર" → "શ્રી ગણેશ ટ્રેક્ટર"
• Odia: "ଶ୍ରୀ ରାମ ଟ୍ରାକ୍ଟର" → "ଶ୍ରୀ ରାମ ଟ୍ରାକ୍ଟର"
• Punjabi: "ਗੁਰੂ ਨਾਨਕ ਟ੍ਰੈਕਟਰ" → "ਗੁਰੂ ਨਾਨਕ ਟ੍ਰੈਕਟਰ"
• Mixed: "Shree राम Tractors" → "Shree राम Tractors"

RULE:
Output text MUST match the original script in the image.
Never translate. Never transliterate. Never normalize.

--------------------------------------------------

TASK:
Analyze the image and extract EXACTLY these four fields.

--------------------------------------------------

dealer_name:
• Entity issuing the invoice / sale contract.
• Can be sub-dealer, franchise, individual, or local agency.
• MUST NOT be tractor manufacturer or brand.
• Prefer name in stamp, header, footer, or near address/GST.
• If the dealer name is written in Hindi or any regional script,
  output it ONLY in that script.
• If only brand/manufacturer name exists → null.

--------------------------------------------------

model_name:
• Tractor model.
• Prefer English alphanumeric models.
• If multiple models listed:
  - Select ONLY the row with handwritten ✓ / tick / circle.
  - Tick must be horizontally aligned and closest.
• If no clear selection → null.

--------------------------------------------------

horse_power:
• Numeric only.
• MUST have literal "HP" text immediately before or after.
• If "HP" is missing → null.
• Reject ranges and inferred values.

--------------------------------------------------

asset_cost:
• Final payable amount only.
• Look for "Grand Total", "Total", or handwritten final amount.
• Ignore subtotals, taxes, line items.
• Digits only (no commas, no currency symbols).
• If ambiguous → null.

--------------------------------------------------

STRICT OUTPUT RULES:
• Output ONLY a single JSON object.
• EXACT keys and order:
  {"dealer_name": "...", "model_name": "...", "horse_power": ..., "asset_cost": ...}
• No extra keys.
• No nesting.
• No explanations.
• No markdown.
• UTF-8 preserved.

--------------------------------------------------

FAILURE POLICY:
If any rule is violated or field is uncertain → output null for that field.
Never guess.
"""
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": processed_path},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        
        # VLM Inference
        text_prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)
        
        generated_ids = self.vlm.generate(**inputs, max_new_tokens=512)
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Parse VLM Output
        try:
            # Extract JSON aggressively
            json_str = output_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "{" in json_str:
                json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
            
            vlm_data = json.loads(json_str)
            
            # Map to final structure
            final_json["fields"]["dealer_name"] = vlm_data.get("dealer_name")
            final_json["fields"]["model_name"] = vlm_data.get("model_name")
            final_json["fields"]["horse_power"] = self._clean_number(
                vlm_data.get("horse_power")
            )
            final_json["fields"]["asset_cost"] = self._clean_number(
                vlm_data.get("asset_cost")
            )
            
            # Calculate confidence
            valid_fields = sum([
                bool(final_json["fields"]["dealer_name"]),
                bool(final_json["fields"]["model_name"]),
                bool(final_json["fields"]["horse_power"]),
                bool(final_json["fields"]["asset_cost"])
            ])
            
            text_conf = valid_fields / 4.0
            visual_conf = max_yolo_conf if max_yolo_conf > 0 else 0.0
            combined_conf = (text_conf * 0.8) + (visual_conf * 0.2)
            final_json["confidence"] = round(combined_conf, 2)
            
        except Exception as e:
            print(f"⚠️  JSON Parse Error: {e}")
            final_json["confidence"] = 0.10
        
        final_json["processing_time_sec"] = round(time.time() - start_time, 2)
        
        return final_json