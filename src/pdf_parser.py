import PyPDF2
import pdfplumber
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class PDFParser:
    def __init__(self):
        # Enhanced section detection patterns with food/recipe focus
        self.section_patterns = [
            # Recipe names and food items (highest priority for food documents)
            r'^\s*([A-Z][a-zA-Z\s]+(?:Salad|Soup|Pasta|Rice|Beans|Vegetables|Casserole|Stir[- ]?Fry|Curry|Bowl))\s*$',
            r'^\s*([A-Z][a-zA-Z\s]*(?:Vegetarian|Vegan|Gluten[- ]?Free)[A-Za-z\s]*)\s*$',
            r'^\s*(Quinoa|Lentil|Chickpea|Tofu|Tempeh|Hummus|Falafel)[A-Za-z\s]*\s*$',
            
            # Recipe sections
            r'^\s*(Ingredients?)\s*:?\s*$',
            r'^\s*(Instructions?|Directions?|Method|Preparation)\s*:?\s*$',
            r'^\s*(Appetizers?|Mains?|Sides?|Desserts?|Beverages?)\s*$',
            
            # ALL CAPS headers (recipes often use these)
            r'^\s*([A-Z][A-Z\s]{3,}[A-Z])\s*$',
            
            # Numbered sections
            r'^\s*(\d+\.?\s+[A-Z][^.!?]+)\s*$',
            r'^\s*(\d+\.\d+\.?\s+[A-Z][^.!?]+)\s*$',
            
            # Letter sections
            r'^\s*([A-Z]\.?\s+[A-Z][^.!?]+)\s*$',
            
            # Title case headers
            r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
            
            # Common recipe/food patterns
            r'^\s*(Breakfast|Lunch|Dinner|Snack|Appetizer|Main|Side|Dessert)\s+[A-Za-z\s]+\s*$',
        ]
        
        # Patterns that indicate NOT a header
        self.non_header_patterns = [
            r'.*[.!?]\s+[A-Z].*',  # Multiple sentences
            r'.*\d+.*\d+.*',       # Multiple numbers (measurements)
            r'.{120,}',            # Very long lines
            r'.*[,;:].*[,;:].*',   # Multiple punctuation marks
            r'^\s*\d+\s*$',        # Just a number
            r'^\s*[^\w\s]+\s*$',   # Just symbols
            r'^\s*[Oo]\s+.*',      # Bullet points starting with 'o'
            r'^\s*[•\-\*]\s+.*',   # Other bullet points
        ]
        
        # Food-specific quality indicators
        self.quality_indicators = {
            'vegetarian_terms': [
                'vegetarian', 'vegan', 'plant-based', 'meatless', 'quinoa', 'lentil', 
                'chickpea', 'tofu', 'tempeh', 'beans', 'legumes', 'nuts', 'seeds',
                'vegetables', 'salad', 'pasta', 'rice', 'grains', 'hummus', 'falafel'
            ],
            'gluten_free_terms': [
                'gluten-free', 'gluten free', 'gf', 'rice', 'quinoa', 'corn', 'potato',
                'vegetables', 'fruits', 'salad', 'beans', 'lentils', 'nuts', 'seeds'
            ],
            'buffet_suitable': [
                'buffet', 'serve warm', 'serve cold', 'room temperature', 'large batch',
                'serves', 'portions', 'casserole', 'tray', 'platter', 'bowl'
            ],
            'non_vegetarian_terms': [
                'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'meat', 'bacon',
                'ham', 'turkey', 'lamb', 'seafood', 'shrimp', 'crab', 'lobster'
            ],
            'high_value_terms': [
                'recipe', 'ingredients', 'instructions', 'preparation', 'cooking',
                'serves', 'portions', 'minutes', 'hours', 'temperature', 'bake',
                'cook', 'simmer', 'sauté', 'mix', 'combine', 'season'
            ]
        }
    
    def parse_pdf(self, pdf_path):
        """Enhanced PDF parsing with food-specific filtering"""
        document = {
            'filename': Path(pdf_path).name,
            'pages': [],
            'sections': [],
            'metadata': {
                'total_pages': 0,
                'extraction_method': 'unknown',
                'quality_score': 0.0,
                'food_type': self._determine_food_type(Path(pdf_path).name)
            }
        }
        
        try:
            # Primary method: pdfplumber
            document = self._parse_with_pdfplumber(pdf_path, document)
            document['metadata']['extraction_method'] = 'pdfplumber'
            
        except Exception as e:
            print(f"Pdfplumber failed for {pdf_path}: {str(e)}")
            try:
                # Fallback: PyPDF2
                document = self._parse_with_pypdf2(pdf_path, document)
                document['metadata']['extraction_method'] = 'pypdf2_fallback'
            except Exception as e2:
                print(f"PyPDF2 fallback also failed for {pdf_path}: {str(e2)}")
                document['metadata']['extraction_method'] = 'failed'
        
        # Post-process and enhance sections with food filtering
        document = self._post_process_document(document)
        
        return document
    
    def _determine_food_type(self, filename):
        """Determine the type of food document based on filename"""
        filename_lower = filename.lower()
        if 'breakfast' in filename_lower:
            return 'breakfast'
        elif 'lunch' in filename_lower:
            return 'lunch'
        elif 'dinner' in filename_lower:
            if 'main' in filename_lower:
                return 'dinner_mains'
            elif 'side' in filename_lower:
                return 'dinner_sides'
            else:
                return 'dinner'
        else:
            return 'general'
    
    def _parse_with_pdfplumber(self, pdf_path, document):
        """Enhanced parsing with pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                document['metadata']['total_pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Enhanced text extraction
                    text = self._extract_text_enhanced(page)
                    
                    if text.strip():  # Only process non-empty pages
                        page_data = {
                            'page_number': page_num,
                            'text': text,
                            'sections': self._extract_sections_enhanced(text, page_num),
                            'char_count': len(text),
                            'word_count': len(text.split())
                        }
                        
                        document['pages'].append(page_data)
                        document['sections'].extend(page_data['sections'])
                
        except Exception as e:
            print(f"Enhanced pdfplumber parsing failed: {e}")
            raise
        
        return document
    
    def _extract_text_enhanced(self, page):
        """Enhanced text extraction with better handling of complex layouts"""
        try:
            # Try multiple extraction strategies
            strategies = [
                lambda p: p.extract_text(x_tolerance=3, y_tolerance=3),
                lambda p: p.extract_text(x_tolerance=1, y_tolerance=1),
                lambda p: p.extract_text(),
            ]
            
            best_text = ""
            max_length = 0
            
            for strategy in strategies:
                try:
                    text = strategy(page) or ""
                    if len(text) > max_length:
                        best_text = text
                        max_length = len(text)
                except:
                    continue
            
            # Clean and normalize the extracted text
            return self._clean_extracted_text(best_text)
            
        except Exception as e:
            print(f"Warning: Enhanced text extraction failed: {e}")
            return page.extract_text() or ""
    
    def _clean_extracted_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        try:
            # Normalize line breaks and spacing
            text = re.sub(r'\r\n?', '\n', text)  # Normalize line endings
            text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive line breaks
            
            # Fix common OCR/extraction issues
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix concatenated words
            text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
            text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', text)  # Join broken words
            
            # Clean up spacing
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
            text = re.sub(r' +\n', '\n', text)   # Remove trailing spaces
            text = re.sub(r'\n +', '\n', text)   # Remove leading spaces
            
            return text.strip()
            
        except Exception:
            return text
    
    def _parse_with_pypdf2(self, pdf_path, document):
        """Enhanced PyPDF2 fallback parser"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                document['metadata']['total_pages'] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        # Enhanced PyPDF2 text extraction
                        text = self._extract_pypdf2_text_enhanced(page)
                        
                        if text.strip():
                            page_data = {
                                'page_number': page_num,
                                'text': text,
                                'sections': self._extract_sections_enhanced(text, page_num),
                                'char_count': len(text),
                                'word_count': len(text.split())
                            }
                            
                            document['pages'].append(page_data)
                            document['sections'].extend(page_data['sections'])
                    except Exception as e:
                        print(f"Warning: Error processing page {page_num}: {e}")
                        continue
                
        except Exception as e:
            print(f"PyPDF2 parsing failed: {e}")
            raise
        
        return document
    
    def _extract_pypdf2_text_enhanced(self, page):
        """Enhanced PyPDF2 text extraction"""
        try:
            # Try different extraction methods
            methods = [
                lambda p: p.extract_text(),
                lambda p: p.extractText() if hasattr(p, 'extractText') else ""
            ]
            
            best_text = ""
            for method in methods:
                try:
                    text = method(page) or ""
                    if len(text) > len(best_text):
                        best_text = text
                except:
                    continue
            
            return self._clean_extracted_text(best_text)
            
        except Exception:
            return ""
    
    def _extract_sections_enhanced(self, text, page_number):
        """Enhanced section extraction with recipe-focused pattern recognition"""
        if not text.strip():
            return []
        
        sections = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Try to identify recipe blocks first
        recipe_sections = self._identify_recipe_blocks(lines, page_number)
        if recipe_sections:
            sections.extend(recipe_sections)
        
        # If no recipe blocks found, use standard section detection
        if not sections:
            sections = self._standard_section_detection(lines, page_number)
        
        # If still no sections, create intelligent chunks
        if not sections and text.strip():
            sections = self._create_intelligent_food_chunks(text, page_number)
        
        return sections
    
    def _identify_recipe_blocks(self, lines, page_number):
        """Identify complete recipe blocks (title + ingredients + instructions)"""
        recipes = []
        current_recipe = None
        current_content = []
        in_ingredients = False
        in_instructions = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if this looks like a recipe title
            if self._is_recipe_title(line):
                # Save previous recipe if it has content
                if current_recipe and current_content:
                    recipe_obj = self._create_recipe_object(
                        current_recipe, current_content, page_number
                    )
                    if recipe_obj:
                        recipes.append(recipe_obj)
                
                # Start new recipe
                current_recipe = line
                current_content = []
                in_ingredients = False
                in_instructions = False
                
            elif 'ingredient' in line_lower:
                in_ingredients = True
                in_instructions = False
                current_content.append(line)
                
            elif any(word in line_lower for word in ['instruction', 'direction', 'method', 'preparation']):
                in_ingredients = False
                in_instructions = True
                current_content.append(line)
                
            else:
                # Add content if we're in a recipe
                if current_recipe:
                    current_content.append(line)
        
        # Add final recipe
        if current_recipe and current_content:
            recipe_obj = self._create_recipe_object(
                current_recipe, current_content, page_number
            )
            if recipe_obj:
                recipes.append(recipe_obj)
        
        return recipes
    
    def _is_recipe_title(self, line):
        """Check if a line looks like a recipe title"""
        if len(line) < 3 or len(line) > 80:
            return False
        
        line_lower = line.lower()
        
        # Check for food-related terms
        food_terms = ['salad', 'soup', 'pasta', 'rice', 'beans', 'vegetables', 
                     'quinoa', 'lentil', 'chickpea', 'tofu', 'casserole', 
                     'stir fry', 'curry', 'bowl', 'wrap', 'sandwich']
        
        has_food_term = any(term in line_lower for term in food_terms)
        
        # Check formatting (title case, all caps, etc.)
        is_title_case = line.istitle()
        is_mostly_caps = sum(1 for c in line if c.isupper()) > len(line) * 0.5
        
        # Check if it's likely a title (not a bullet point or instruction)
        not_bullet = not line.startswith(('•', '-', '*', 'o '))
        not_instruction = not any(word in line_lower for word in ['add', 'mix', 'cook', 'bake', 'serve'])
        
        return (has_food_term or is_title_case or is_mostly_caps) and not_bullet and not_instruction
    
    def _create_recipe_object(self, title, content_lines, page_number):
        """Create a recipe object with food-specific metadata"""
        if not content_lines:
            return None
        
        content = ' '.join(content_lines)
        word_count = len(content.split())
        
        # Quality filtering - skip very short recipes
        if word_count < 20:
            return None
        
        # Calculate food-specific quality score
        quality_score = self._calculate_food_quality(title, content)
        
        # Check vegetarian/vegan status
        is_vegetarian = self._is_vegetarian(content)
        is_gluten_free = self._is_gluten_free(content)
        is_buffet_suitable = self._is_buffet_suitable(content)
        
        section = {
            'title': self._clean_section_title(title),
            'content': self._clean_section_content(content),
            'page_number': page_number,
            'word_count': word_count,
            'char_count': len(content),
            'quality_score': quality_score,
            'is_vegetarian': is_vegetarian,
            'is_gluten_free': is_gluten_free,
            'is_buffet_suitable': is_buffet_suitable,
            'food_type': 'recipe'
        }
        
        return section
    
    def _calculate_food_quality(self, title, content):
        """Calculate quality score specific to food content"""
        if not content:
            return 0.0
        
        try:
            score = 0.0
            content_lower = content.lower()
            title_lower = title.lower() if title else ""
            
            # Base score for having both title and content
            if title and len(content) > 50:
                score += 0.3
            
            # Recipe completeness indicators
            has_ingredients = 'ingredient' in content_lower
            has_instructions = any(word in content_lower for word in ['instruction', 'direction', 'cook', 'bake', 'mix'])
            has_measurements = bool(re.search(r'\d+\s*(cup|tablespoon|teaspoon|pound|ounce|gram)', content_lower))
            
            if has_ingredients:
                score += 0.2
            if has_instructions:
                score += 0.2
            if has_measurements:
                score += 0.1
            
            # Vegetarian/buffet bonus
            if self._is_vegetarian(content):
                score += 0.15
            if self._is_buffet_suitable(content):
                score += 0.1
            
            # Food-specific terms bonus
            food_term_count = sum(1 for term in self.quality_indicators['high_value_terms'] 
                                if term in content_lower)
            score += min(0.1, food_term_count * 0.02)
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _is_vegetarian(self, content):
        """Check if content appears to be vegetarian"""
        content_lower = content.lower()
        
        # Check for non-vegetarian terms
        has_meat = any(term in content_lower for term in self.quality_indicators['non_vegetarian_terms'])
        if has_meat:
            return False
        
        # Check for vegetarian indicators
        has_veg_terms = any(term in content_lower for term in self.quality_indicators['vegetarian_terms'])
        
        return has_veg_terms or not has_meat
    
    def _is_gluten_free(self, content):
        """Check if content appears to be gluten-free"""
        content_lower = content.lower()
        
        # Check for explicit gluten-free mention
        if any(term in content_lower for term in ['gluten-free', 'gluten free', 'gf']):
            return True
        
        # Check for gluten-containing ingredients
        gluten_terms = ['wheat', 'flour', 'bread', 'pasta', 'noodles', 'soy sauce', 'beer']
        has_gluten = any(term in content_lower for term in gluten_terms)
        
        if has_gluten:
            return False
        
        # Check for naturally gluten-free ingredients
        gf_terms = self.quality_indicators['gluten_free_terms']
        has_gf_terms = any(term in content_lower for term in gf_terms)
        
        return has_gf_terms
    
    def _is_buffet_suitable(self, content):
        """Check if content is suitable for buffet service"""
        content_lower = content.lower()
        
        # Check for buffet indicators
        buffet_indicators = self.quality_indicators['buffet_suitable']
        return any(term in content_lower for term in buffet_indicators)
    
    def _standard_section_detection(self, lines, page_number):
        """Standard section detection for non-recipe content"""
        sections = []
        current_section = None
        section_content = []
        last_header_confidence = 0.0
        
        for i, line in enumerate(lines):
            # Enhanced header detection
            is_header, confidence, header_text = self._is_section_header_enhanced(line, i, lines)
            
            if is_header and confidence > 0.5:  # Lower threshold for food documents
                # Save previous section if it has substantial content
                if current_section and section_content:
                    section_obj = self._create_section_object(
                        current_section, section_content, page_number, last_header_confidence
                    )
                    if section_obj:  # Only add if it meets quality criteria
                        sections.append(section_obj)
                
                # Start new section
                current_section = header_text
                section_content = []
                last_header_confidence = confidence
                
            else:
                # Add to current section content
                if current_section:
                    section_content.append(line)
                elif not sections:  # No sections detected yet, start collecting content
                    if not current_section:
                        current_section = self._generate_section_title(line, i)
                    section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            section_obj = self._create_section_object(
                current_section, section_content, page_number, last_header_confidence
            )
            if section_obj:
                sections.append(section_obj)
        
        return sections
    
    def _is_section_header_enhanced(self, line, line_index, all_lines):
        """Enhanced header detection with confidence scoring"""
        if not line or len(line) < 3:
            return False, 0.0, line
        
        confidence = 0.0
        
        # Check against non-header patterns first
        for pattern in self.non_header_patterns:
            if re.match(pattern, line):
                return False, 0.0, line
        
        # Check against header patterns
        best_match = None
        for i, pattern in enumerate(self.section_patterns):
            try:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Weight by pattern importance
                    pattern_confidence = 1.0 - (i * 0.05)
                    if pattern_confidence > confidence:
                        confidence = pattern_confidence
                        best_match = match.group(1) if match.groups() else line
            except:
                continue
        
        if confidence > 0:
            # Additional confidence factors for food documents
            
            # Length factor
            if 5 <= len(line) <= 60:  # Shorter for food titles
                confidence += 0.1
            elif len(line) > 80:
                confidence -= 0.2
            
            # Food-specific bonuses
            line_lower = line.lower()
            if any(term in line_lower for term in self.quality_indicators['vegetarian_terms']):
                confidence += 0.15
            if any(term in line_lower for term in ['recipe', 'ingredients', 'instructions']):
                confidence += 0.1
            
            # Formatting factors
            if line.isupper() and len(line) > 3:
                confidence += 0.1
            elif line.istitle():
                confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        return confidence > 0.3, confidence, best_match or line
    
    def _generate_section_title(self, first_line, position):
        """Generate a section title when none detected"""
        if len(first_line) <= 50:
            return first_line
        
        # Take first few words
        words = first_line.split()[:6]
        title = ' '.join(words)
        
        if len(title) < 50:
            return title
        else:
            return title[:47] + "..."
    
    def _create_section_object(self, title, content_lines, page_number, confidence):
        """Create a section object with enhanced metadata"""
        if not content_lines:
            return None
        
        content = ' '.join(content_lines)
        word_count = len(content.split())
        
        # Quality filtering - skip very short sections
        if word_count < 15:
            return None
        
        # Calculate content quality score
        quality_score = self._calculate_food_quality(title, content)
        
        section = {
            'title': self._clean_section_title(title),
            'content': self._clean_section_content(content),
            'page_number': page_number,
            'word_count': word_count,
            'char_count': len(content),
            'sentence_count': len([s for s in re.split(r'[.!?]+', content) if len(s.strip()) > 5]),
            'header_confidence': round(confidence, 3),
            'quality_score': quality_score,
            'is_vegetarian': self._is_vegetarian(content),
            'is_gluten_free': self._is_gluten_free(content),
            'is_buffet_suitable': self._is_buffet_suitable(content)
        }
        
        return section
    
    def _clean_section_title(self, title):
        """Enhanced title cleaning for food content"""
        if not title:
            return "Recipe Section"
        
        try:
            cleaned = title.strip()
            
            # Remove numbering and formatting
            cleaned = re.sub(r'^[•\-\*\d+\.\)\]\s]+', '', cleaned)
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            cleaned = re.sub(r'^[IVX]+[\.\)]\s*', '', cleaned, flags=re.IGNORECASE)
            
            # Remove formatting characters
            cleaned = re.sub(r'[*_]+', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Capitalize properly
            if cleaned and len(cleaned) > 1:
                if cleaned.islower():
                    cleaned = cleaned.title()
                elif not cleaned[0].isupper():
                    cleaned = cleaned[0].upper() + cleaned[1:]
            
            # Length limit
            if len(cleaned) > 80:  # Shorter for food titles
                words = cleaned.split()
                if len(words) > 10:
                    cleaned = ' '.join(words[:10]) + "..."
                else:
                    cleaned = cleaned[:77] + "..."
            
            return cleaned if len(cleaned) >= 3 else "Recipe Section"
            
        except Exception:
            return "Recipe Section"
    
    def _clean_section_content(self, content):
        """Enhanced content cleaning for food content"""
        if not content:
            return ""
        
        try:
            # Basic normalization
            cleaned = re.sub(r'\s+', ' ', content.strip())
            
            # Fix common extraction artifacts
            cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
            cleaned = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned)
            
            # Remove excessive punctuation
            cleaned = re.sub(r'\.{3,}', '...', cleaned)
            cleaned = re.sub(r'[\-_]{3,}', '---', cleaned)
            
            return cleaned
            
        except Exception:
            return content
    
    def _create_intelligent_food_chunks(self, text, page_number, target_chunk_size=300):
        """Create intelligent food-focused chunks"""
        try:
            # Split into logical food sections
            chunks = []
            
            # Look for recipe patterns in the text
            recipe_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s]*(?:Salad|Soup|Pasta|Rice|Beans|Vegetables))', text)
            
            for i, block in enumerate(recipe_blocks):
                if len(block.strip()) < 50:
                    continue
                
                # Generate title from first line or content
                lines = block.strip().split('\n')
                title = self._generate_food_chunk_title(lines[0], i + 1)
                
                chunk = {
                    'title': title,
                    'content': block.strip(),
                    'page_number': page_number,
                    'word_count': len(block.split()),
                    'quality_score': self._calculate_food_quality(title, block),
                    'is_vegetarian': self._is_vegetarian(block),
                    'is_gluten_free': self._is_gluten_free(block),
                    'chunk_type': 'food_based'
                }
                chunks.append(chunk)
            
            return chunks if chunks else self._create_word_chunks(text, page_number)
            
        except Exception as e:
            print(f"Warning: Food chunking failed: {e}")
            return self._create_word_chunks(text, page_number)
    
    def _generate_food_chunk_title(self, first_line, chunk_num):
        """Generate food-focused title for content chunks"""
        try:
            if len(first_line) <= 50 and any(term in first_line.lower() for term in 
                                           ['salad', 'soup', 'pasta', 'rice', 'beans', 'vegetables']):
                return first_line.strip()
            
            # Extract food-related words
            words = first_line.split()
            food_words = []
            
            for word in words[:8]:
                word_lower = word.lower()
                if any(term in word_lower for term in self.quality_indicators['vegetarian_terms']):
                    food_words.append(word)
            
            if food_words:
                title = ' '.join(food_words[:4])
            else:
                title = ' '.join(words[:5])
            
            # Ensure reasonable length
            if len(title) > 50:
                title = title[:47] + "..."
            
            return title if title else f"Food Section {chunk_num}"
            
        except Exception:
            return f"Food Section {chunk_num}"
    
    def _create_word_chunks(self, text, page_number, chunk_size=400):
        """Fallback: create simple word-based chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Create title from first few words
            title_words = chunk_words[:6]
            title = ' '.join(title_words)
            if len(title) > 50:
                title = title[:47] + "..."
            
            chunks.append({
                'title': title,
                'content': chunk_text,
                'page_number': page_number,
                'word_count': len(chunk_words),
                'quality_score': 0.5,  # Default quality for word chunks
                'is_vegetarian': self._is_vegetarian(chunk_text),
                'is_gluten_free': self._is_gluten_free(chunk_text),
                'chunk_type': 'word_based'
            })
        
        return chunks
    
    def _post_process_document(self, document):
        """Post-process document with food-specific filtering and enhancement"""
        try:
            sections = document.get('sections', [])
            
            if sections:
                # Filter out non-vegetarian sections if this is for vegetarian buffet
                vegetarian_sections = []
                for section in sections:
                    # Keep vegetarian items or items that don't contain meat
                    if (section.get('is_vegetarian', True) and 
                        not any(term in section.get('content', '').lower() 
                               for term in self.quality_indicators['non_vegetarian_terms'])):
                        vegetarian_sections.append(section)
                
                # Calculate document-level quality metrics
                if vegetarian_sections:
                    quality_scores = [s.get('quality_score', 0.5) for s in vegetarian_sections]
                    word_counts = [s.get('word_count', 0) for s in vegetarian_sections]
                    vegetarian_count = len([s for s in vegetarian_sections if s.get('is_vegetarian', False)])
                    gluten_free_count = len([s for s in vegetarian_sections if s.get('is_gluten_free', False)])
                    buffet_suitable_count = len([s for s in vegetarian_sections if s.get('is_buffet_suitable', False)])
                    
                    document['metadata'].update({
                        'total_sections': len(vegetarian_sections),
                        'avg_quality_score': round(sum(quality_scores) / len(quality_scores), 3),
                        'total_words': sum(word_counts),
                        'high_quality_sections': len([s for s in vegetarian_sections if s.get('quality_score', 0) > 0.7]),
                        'vegetarian_sections': vegetarian_count,
                        'gluten_free_sections': gluten_free_count,
                        'buffet_suitable_sections': buffet_suitable_count
                    })
                    
                    # Sort sections by relevance for vegetarian buffet
                    document['sections'] = sorted(vegetarian_sections, key=lambda x: (
                        -x.get('is_buffet_suitable', 0),  # Buffet suitable first
                        -x.get('is_gluten_free', 0),      # Then gluten-free
                        -x.get('quality_score', 0),       # Then by quality
                        x.get('page_number', 0)           # Finally by page order
                    ))
                    
                    # Add section rankings based on vegetarian buffet relevance
                    for i, section in enumerate(document['sections']):
                        section['section_rank'] = i + 1
                        section['buffet_relevance_score'] = self._calculate_buffet_relevance(section)
                else:
                    document['sections'] = []
            
            if not document['sections']:
                document['metadata'].update({
                    'total_sections': 0,
                    'avg_quality_score': 0.0,
                    'total_words': 0,
                    'high_quality_sections': 0,
                    'vegetarian_sections': 0,
                    'gluten_free_sections': 0,
                    'buffet_suitable_sections': 0
                })
            
            return document
            
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
            return document
    
    def _calculate_buffet_relevance(self, section):
        """Calculate how relevant a section is for a vegetarian buffet"""
        score = 0.0
        content = section.get('content', '').lower()
        title = section.get('title', '').lower()
        
        # Base quality score
        score += section.get('quality_score', 0.5) * 0.3
        
        # Vegetarian bonus
        if section.get('is_vegetarian', False):
            score += 0.25
        
        # Gluten-free bonus
        if section.get('is_gluten_free', False):
            score += 0.2
        
        # Buffet suitable bonus
        if section.get('is_buffet_suitable', False):
            score += 0.15
        
        # Food type relevance
        food_terms = ['salad', 'vegetables', 'rice', 'pasta', 'beans', 'quinoa', 'lentil']
        if any(term in title or term in content for term in food_terms):
            score += 0.1
        
        return round(min(1.0, score), 3)