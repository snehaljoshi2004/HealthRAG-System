# cardiology_data.py
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import json

# ============================================
# COMPREHENSIVE CARDIOLOGY DATASET
# ============================================

CARDIOLOGY_QA = [
    # ========== PARADIGM-HF / Heart Failure ==========
    {
        "question": "In the PARADIGM-HF study, LCZ696 was used, which is a combination of: A. a neprilysin inhibitor and an ARB",
        "answer": """**LCZ696 (sacubitril/valsartan)** is a first-in-class angiotensin receptor-neprilysin inhibitor (ARNI).

## Composition:
- **Sacubitril** - a neprilysin inhibitor (increases natriuretic peptides)
- **Valsartan** - an angiotensin II receptor blocker (ARB)

## PARADIGM-HF Trial (2014) Key Results:
- **Primary outcome**: 20% reduction in cardiovascular death or HF hospitalization (HR 0.80, p<0.001)
- **NNT**: 21 patients to prevent one primary outcome event
- **CV death**: 20% reduction (HR 0.80)
- **HF hospitalization**: 21% reduction (HR 0.79)
- **All-cause mortality**: 16% reduction (HR 0.84)

## Current Guidelines:
- **ACC/AHA/HFSA**: Class I recommendation for HFrEF (NYHA II-III) to replace ACE inhibitor/ARB
- **ESC**: Class I recommendation for HFrEF (LVEF ≤40%) who remain symptomatic on ACE inhibitor/ARB, beta-blocker, and MRA

## Mechanism:
Sacubitril inhibits neprilysin, increasing levels of natriuretic peptides (ANP, BNP), bradykinin, and adrenomedullin, leading to vasodilation, natriuresis, and reduced fibrosis. Valsartan blocks AT1 receptors, reducing vasoconstriction and aldosterone secretion.""",
        "source": "PARADIGM-HF Trial, NEJM 2014; McMurray JJV et al.; 2022 AHA/ACC/HFSA Heart Failure Guidelines"
    },
    
    # ========== Digitalis/Verapamil Contraindications ==========
    {
        "question": "The use of digitalis and verapamil is absolutely contraindicated in:",
        "answer": """**Digitalis and verapamil are absolutely contraindicated in patients with:**

## ✅ Wolff-Parkinson-White (WPW) Syndrome with Atrial Fibrillation

### Why? 
Both drugs slow conduction through the **AV node** but can **accelerate conduction through the accessory pathway** (Kent bundle), potentially leading to:
- **Extremely rapid ventricular rates** (>300 bpm)
- **Degeneration into ventricular fibrillation**
- **Sudden cardiac death**

### Mechanism:
- **Digitalis**: Increases vagal tone → slows AV node conduction
- **Verapamil**: Directly blocks calcium channels → slows AV node conduction
- In WPW, the accessory pathway lacks the decremental conduction properties of the AV node
- AV nodal blockade shifts conduction to the accessory pathway → rapid ventricular response

## Other Important Contraindications:

### Digitalis:
- **AV block** (2nd degree Mobitz II, complete heart block)
- **Ventricular tachycardia/fibrillation**
- **Hypertrophic cardiomyopathy** (obstructive form)
- **Hypokalemia** (increases toxicity)
- **Hypersensitivity**

### Verapamil:
- **Severe hypotension** (SBP <90 mmHg)
- **Cardiogenic shock**
- **Sick sinus syndrome** (without pacemaker)
- **2nd/3rd degree AV block** (without pacemaker)
- **Concurrent IV beta-blockers**

## Clinical Pearl:
In WPW with wide-complex tachycardia or pre-excited AF, **avoid AV nodal blocking agents** (adenosine, verapamil, diltiazem, digoxin). Use:
- **Procainamide** (Class IA) or
- **Amiodarone** (Class III) or
- **Ibutilide** (Class III)
- **Electrical cardioversion** if unstable

## ECG Findings in WPW:
- Short PR interval (<120 ms)
- Delta wave (slurred upstroke of QRS)
- Wide QRS (>110 ms)
- Secondary ST-T wave changes""",
        "source": "2023 ACC/AHA/ESC Guidelines for AF Management; Braunwald's Heart Disease 12th ed."
    },
    
    # ========== STEMI - COMPREHENSIVE ==========
    {
        "question": "A 76-year-old man presents with sudden chest pain and shortness of breath. ECG shows ST-elevation in leads V1-V4. What is the diagnosis and management?",
        "answer": """## DIAGNOSIS: **Acute Anterior ST-Elevation Myocardial Infarction (STEMI)**

### ECG Localization:
- **V1-V4 ST elevation** → **Left Anterior Descending (LAD) artery** occlusion
- **Anterior wall MI** involves the left ventricular anterior wall and septum
- **Complications risk**: Higher risk of heart failure, cardiogenic shock, and arrhythmias

---

## IMMEDIATE MANAGEMENT (Door-to-Balloon Time <90 minutes)

### 1. **MONA** (if no contraindications):
| Drug | Dose | Purpose |
|------|------|---------|
| **M**orphine | 2-4 mg IV | Pain relief, reduces sympathetic tone |
| **O**xygen | 2-4 L/min if O2 sat <90% | Avoid hyperoxia if sat >90% |
| **N**itroglycerin | 0.4 mg SL q5min x3, then IV | Vasodilation, pain relief |
| **A**spirin | 325 mg chewed | Antiplatelet, reduces mortality by 23% |

### 2. **Dual Antiplatelet Therapy (DAPT)**:
- **Aspirin** 325 mg load, then 81 mg daily
- **P2Y12 inhibitor** (choose one):
  - **Ticagrelor** 180 mg load, then 90 mg BID (preferred)
  - **Prasugrel** 60 mg load, then 10 mg daily (if no prior CVA, age <75, weight >60kg)
  - Clopidogrel 600 mg load, then 75 mg daily (if above contraindicated)

### 3. **Anticoagulation**:
| Agent | Dose | Duration |
|-------|------|----------|
| Unfractionated Heparin | 60 U/kg bolus (max 4000U), then 12 U/kg/hr (max 1000U/hr) | During PCI |
| Enoxaparin | 0.5 mg/kg IV bolus, then 1 mg/kg SC BID | Up to 8 days |
| Bivalirudin | 0.75 mg/kg bolus, then 1.75 mg/kg/hr | During PCI |

### 4. **Reperfusion Strategy**:
**Primary PCI** (preferred if within 120 minutes):
- Door-to-balloon time **<90 minutes**
- Radial access preferred
- Drug-eluting stent preferred over bare-metal

**Fibrinolysis** (if PCI not available within 120 minutes):
- Door-to-needle time **<30 minutes**
- Agents: Tenecteplase (TNK), Alteplase (tPA)
- Transfer to PCI-capable center within 3-24 hours

---

## ADJUNCTIVE THERAPY:

| Drug Class | Examples | Initiation Timing |
|------------|----------|-------------------|
| **Beta-blocker** | Metoprolol 25-50 mg BID | Within 24 hours (if no contraindications) |
| **ACE Inhibitor/ARB** | Lisinopril 5-10 mg daily | Within 24 hours (especially anterior MI) |
| **Statin** | Atorvastatin 80 mg daily | High-intensity statin immediately |
| **Aldosterone Antagonist** | Eplerenone 25 mg daily | If LVEF ≤40% with HF or diabetes |

---

## COMPLICATIONS TO MONITOR:

### Early Complications (<48 hours):
| Complication | Presentation | Management |
|--------------|--------------|-------------|
| **Arrhythmias** | VT/VF, AF | Amiodarone, cardioversion |
| **Cardiogenic shock** | Hypotension, pulmonary edema | Inotropes, IABP, LVAD |
| **Mechanical complications** | New murmur, hypotension | Emergent surgery |

### Late Complications (days-weeks):
| Complication | Presentation | Diagnosis | Management |
|--------------|--------------|-----------|------------|
| **Ventricular septal rupture** | Harsh holosystolic murmur, shock | Echo, Swan-Ganz (O2 step-up) | Surgical repair |
| **Papillary muscle rupture** | Acute mitral regurgitation, pulmonary edema | Echo | Surgical repair |
| **Free wall rupture** | Tamponade, PEA arrest | Echo | Emergent surgery |
| **Pericarditis** | Chest pain, pericardial friction rub | ECG (diffuse ST elevation) | NSAIDs, colchicine |

---

## DISCHARGE MEDICATIONS:
1. **Aspirin** 81 mg daily (lifelong)
2. **P2Y12 inhibitor** (ticagrelor/prasugrel) for 12 months
3. **Beta-blocker** (lifelong if LVEF ≤40%)
4. **ACE Inhibitor/ARB** (lifelong)
5. **High-intensity statin** (lifelong)
6. **Aldosterone antagonist** if LVEF ≤40%

---

## PROGNOSIS:
- In-hospital mortality: **4-12%** (higher with anterior MI)
- 30-day mortality: **5-15%**
- 1-year mortality: **10-20%**
- LVEF recovery depends on time to reperfusion""",
        "source": "2021 ACC/AHA/SCAI STEMI Guidelines; 2023 ESC Guidelines for Management of Acute MI; Braunwald's Heart Disease"
    },
    
    # ========== ADDITIONAL STEMI VARIANTS ==========
    {
        "question": "What are the ECG criteria for diagnosing STEMI and how do you localize the culprit artery?",
        "answer": """## STEMI ECG DIAGNOSTIC CRITERIA

### Standard Criteria (≥40 years):
- **ST elevation at J-point** in **2 contiguous leads**:
  - **V2-V3**: ≥2 mm in men ≥40, ≥2.5 mm in men <40, ≥1.5 mm in women
  - **All other leads**: ≥1 mm

### STEMI Localization by ECG:

| Territory | Leads | Artery | Walls Involved |
|-----------|-------|--------|----------------|
| **Anterior** | V1-V4 | LAD | Anterior wall, septum |
| **Lateral** | I, aVL, V5-V6 | LCx or diagonal branches | Lateral wall |
| **Inferior** | II, III, aVF | RCA (80%) or LCx (20%) | Inferior wall |
| **Posterior** | V7-V9, reciprocal changes V1-V3 | LCx or RCA | Posterior wall |
| **Right ventricular** | V4R | Proximal RCA | RV free wall |

### Reciprocal Changes:
- **Anterior STEMI**: Reciprocal ST depression in inferior leads (II, III, aVF)
- **Inferior STEMI**: Reciprocal ST depression in lateral leads (I, aVL)
- **Posterior STEMI**: ST depression V1-V3 with tall R waves

### High-Risk Features:
- **Extensive anterior ST elevation** → LAD proximal occlusion
- **ST elevation in aVR** → Left main or multi-vessel disease
- **Wellens' syndrome** (deep TWI V2-V3) → Critical proximal LAD stenosis
- **De Winter pattern** (up-sloping ST depression with tall T waves) → LAD occlusion""",
        "source": "2021 ACC/AHA/SCAI STEMI Guidelines; Chou's Electrocardiography in Clinical Practice"
    },
    
    # ========== HEART RATE EFFECTS ON VALVULAR LESIONS ==========
    {
        "question": "Choose the characteristic effect of altered heart rate on valvular heart lesions (two correct answers):",
        "answer": """## EFFECTS OF HEART RATE ON VALVULAR LESIONS

### ✅ CORRECT STATEMENTS:

**1. Increased heart rate decreases the duration of diastolic murmurs**
   - **Rationale**: Tachycardia shortens diastole proportionally more than systole
   - **Affected murmurs**: 
     - Mitral Stenosis (MS) - diastolic rumble
     - Aortic Regurgitation (AR) - early diastolic decrescendo
     - Tricuspid Stenosis (TS) - diastolic murmur
   - **Clinical significance**: In MS, atrial fibrillation with rapid ventricular rate may make the diastolic murmur inaudible

**2. Decreased heart rate increases the intensity of diastolic murmurs**
   - **Rationale**: Bradycardia prolongs diastole, allowing more time for pressure gradients
   - **Affected murmurs**:
     - Mitral Stenosis - longer diastolic filling time increases gradient
     - Aortic Regurgitation - more time for regurgitant flow
   - **Clinical significance**: In MS, prolonged diastole (e.g., after a PVC or in complete heart block) allows full transmission of left atrial-LV pressure gradient

### ❌ INCORRECT STATEMENTS (Common Misconceptions):

- **"Increased heart rate increases systolic murmur intensity"** → FALSE
  - Systolic murmurs (AS, MR, VSD) are less affected by heart rate changes
  - However, conditions with dynamic obstruction (HCM) may worsen with increased contractility

- **"Heart rate affects all murmurs equally"** → FALSE
  - Diastolic murmurs are rate-sensitive (dependent on diastolic duration)
  - Continuous murmurs (PDA) are affected by both systole and diastole

### PHYSIOLOGICAL BASIS:

| Murmur Type | Affected by Rate? | Mechanism |
|-------------|-------------------|-----------|
| Systolic (AS, MR) | Minimal | Systolic duration relatively fixed |
| Diastolic (MS, AR) | YES | Diastolic duration highly variable |
| Continuous (PDA) | Moderate | Both systolic and diastolic components |

### CLINICAL APPLICATIONS:

**Mitral Stenosis**:
- Tachycardia → shorter diastole → decreased murmur duration but increased gradient due to reduced filling time
- Bradycardia → longer diastole → increased murmur duration and intensity

**Aortic Regurgitation**:
- Bradycardia → more time for regurgitation → louder, longer diastolic murmur
- Tachycardia → less regurgitant volume per beat → softer murmur

**Austin Flint Murmur** (AR with MS-like rumble):
- Best heard with patient leaning forward in end-expiration
- May disappear with tachycardia due to shortened diastole""",
        "source": "Braunwald's Heart Disease 12th ed.; Hurst's The Heart; ACC/AHA Valvular Heart Disease Guidelines"
    },
    
    # ========== ADDITIONAL HF CONTENT ==========
    {
        "question": "What is the classification system for heart failure?",
        "answer": """## HEART FAILURE CLASSIFICATION SYSTEMS

### 1. **NYHA Functional Classification** (Symptoms-based)

| Class | Description |
|-------|-------------|
| **Class I** | No limitation of physical activity. Ordinary activity does not cause symptoms. |
| **Class II** | Slight limitation. Comfortable at rest, but ordinary activity causes symptoms. |
| **Class III** | Marked limitation. Comfortable at rest, but less than ordinary activity causes symptoms. |
| **Class IV** | Unable to carry out any physical activity without symptoms. Symptoms at rest. |

### 2. **ACC/AHA Stages of HF** (Structural + Symptoms)

| Stage | Description | Examples |
|-------|-------------|----------|
| **Stage A** | At high risk for HF but no structural disease | Hypertension, diabetes, CAD, family history |
| **Stage B** | Structural heart disease but no symptoms | LVH, low EF, prior MI, valvular disease |
| **Stage C** | Structural disease with prior/current symptoms | Dyspnea, fatigue, reduced exercise tolerance |
| **Stage D** | Refractory HF requiring special interventions | Recurrent hospitalizations, inotrope-dependent, LVAD, transplant |

### 3. **HF Classification by LVEF**

| Type | LVEF | Characteristics |
|------|------|-----------------|
| **HFrEF** (Heart Failure with reduced Ejection Fraction) | ≤40% | Systolic dysfunction, clinical trial evidence for GDMT |
| **HFimpEF** (Heart Failure with improved EF) | >40% with prior ≤40% | Previously HFrEF that improved |
| **HFmrEF** (Heart Failure with mildly reduced EF) | 41-49% | Borderline, treat similar to HFrEF |
| **HFpEF** (Heart Failure with preserved EF) | ≥50% | Diastolic dysfunction, limited treatment options |

### 4. **Killip Classification** (Acute MI setting)

| Class | Description | Mortality |
|-------|-------------|-----------|
| **Class I** | No heart failure | 6% |
| **Class II** | Rales, S3, elevated JVP | 17% |
| **Class III** | Acute pulmonary edema | 38% |
| **Class IV** | Cardiogenic shock | 81% |

### 5. **Forrester Classification** (Hemodynamic)

| Subset | Hemodynamics | PCWP | CI | Management |
|--------|--------------|------|-----|------------|
| I | Warm and dry | <18 | >2.2 | Observation |
| II | Warm and wet | >18 | >2.2 | Diuresis, vasodilators |
| III | Cold and dry | <18 | <2.2 | Volume resuscitation |
| IV | Cold and wet | >18 | <2.2 | Inotropes, vasodilators, diuresis |

## GUIDELINE-DIRECTED MEDICAL THERAPY (GDMT) by Stage:

**Stage A**: Risk factor modification (BP control, statins, lifestyle)
**Stage B**: ACE inhibitor/ARB, beta-blocker if EF reduced
**Stage C**: 
- **HFrEF**: ARNI/ACE/ARB + Beta-blocker + MRA + SGLT2 inhibitor
- **HFpEF**: Diuretics, SGLT2 inhibitor, treat comorbidities
**Stage D**: Advanced therapies (LVAD, transplant, hospice)""",
        "source": "2022 AHA/ACC/HFSA Heart Failure Guidelines; 2021 ESC Heart Failure Guidelines"
    },
    
    # ========== ANTIARRHYTHMIC DRUGS ==========
    {
        "question": "What is the Vaughan Williams classification of antiarrhythmic drugs?",
        "answer": """## VAUGHAN WILLIAMS CLASSIFICATION OF ANTIARRHYTHMIC DRUGS

### Class I: **Sodium Channel Blockers** (Na+ channel blockade)

| Subclass | Drugs | Effect on AP | ECG Changes | Indications |
|----------|-------|--------------|-------------|-------------|
| **IA** | Quinidine, Procainamide, Disopyramide | Moderate blockade, prolongs AP | ↑ QRS, ↑ QT | AF, VT, WPW |
| **IB** | Lidocaine, Mexiletine, Phenytoin | Weak blockade, shortens AP | Minimal | VT, post-MI |
| **IC** | Flecainide, Propafenone | Strong blockade, minimal AP effect | ↑↑ QRS | AF (no structural HD) |

### Class II: **Beta-Blockers** (β-adrenergic blockade)
- **Drugs**: Metoprolol, Carvedilol, Bisoprolol, Propranolol, Esmolol, Atenolol
- **Effect**: ↓ SA node automaticity, ↓ AV node conduction
- **ECG**: ↓ HR, ↑ PR interval
- **Indications**: Rate control in AF, post-MI, HFrEF, VT storm

### Class III: **Potassium Channel Blockers** (K+ channel blockade, prolong repolarization)

| Drug | Mechanism | ECG | Indications | Toxicity |
|------|-----------|-----|-------------|----------|
| **Amiodarone** | Multi-channel blocker | ↑ QT | AF, VT, VF | Pulmonary, thyroid, liver |
| **Sotalol** | β-blocker + K+ blocker | ↑ QT | AF, VT | Torsades, bradycardia |
| **Dofetilide** | Pure K+ blocker | ↑↑ QT | AF | Torsades (in-hospital start) |
| **Ibutilide** | K+ blocker | ↑ QT | Acute AF cardioversion | Torsades |
| **Dronedarone** | Multi-channel | ↑ QT | AF (not permanent) | Liver toxicity |

### Class IV: **Calcium Channel Blockers** (Ca++ channel blockade)
- **Drugs**: Verapamil, Diltiazem
- **Effect**: ↓ SA node, ↓ AV node conduction
- **ECG**: ↓ HR, ↑ PR interval
- **Indications**: Rate control in AF, PSVT
- **Contraindications**: Wide complex tachycardia, WPW with AF

### Other Antiarrhythmics:

| Drug | Class | Mechanism | Use |
|------|-------|-----------|-----|
| **Adenosine** | Unclassified | AV node blockade via A1 receptors | PSVT |
| **Digoxin** | Unclassified | Vagomimetic, Na/K-ATPase inhibition | Rate control in AF + HF |
| **Magnesium** | Unclassified | Membrane stabilization | Torsades, post-MI |
| **Ivabradine** | Unclassified | I(f) channel blocker | HFrEF, inappropriate sinus tachycardia |

## CLINICAL PEARLS:

### Amiodarone Side Effects (remember: "Pulmonary, Thyroid, Liver, Cornea, Blue skin, Neuropathy"):
- Pulmonary fibrosis (most serious, 5-10%)
- Thyroid dysfunction (hyper or hypo)
- Liver toxicity
- Corneal microdeposits (reversible)
- Blue-gray skin discoloration
- Peripheral neuropathy

### Contraindications:
- **Class IC drugs** are contraindicated in structural heart disease (CAST trial)
- **Class III drugs** should not be combined with other QT-prolonging drugs
- **Verapamil/diltiazem** contraindicated in wide complex tachycardia of unknown origin
- **Sotalol** contraindicated in asthma (due to β-blockade) and prolonged QT

### Drug Interactions:
- Amiodarone inhibits CYP enzymes → increases levels of warfarin, digoxin, statins
- Verapamil increases digoxin levels
- Macrolides and fluoroquinolones increase QT prolongation risk""",
        "source": "Goodman & Gilman's Pharmacology; 2023 ACC/AHA/ESC AF Guidelines"
    }
]

# ============================================
# ADDITIONAL CARDIOLOGY KNOWLEDGE
# ============================================

ADDITIONAL_CARDIOLOGY = [
    {
        "topic": "Heart Failure Classification",
        "content": """**NYHA Functional Classification:**
- Class I: No limitation of physical activity
- Class II: Slight limitation, comfortable at rest
- Class III: Marked limitation, comfortable only at rest
- Class IV: Symptoms at rest

**ACC/AHA Stages of HF:**
- Stage A: At high risk but no structural disease
- Stage B: Structural disease but no symptoms
- Stage C: Structural disease with symptoms
- Stage D: Refractory HF requiring special interventions

**HF by EF:**
- HFrEF: EF ≤40%
- HFmrEF: EF 41-49%
- HFpEF: EF ≥50%""",
        "source": "2022 AHA/ACC/HFSA Heart Failure Guidelines"
    },
    
    {
        "topic": "Coronary Artery Territories",
        "content": """**LAD (Left Anterior Descending):**
- Anterior wall (V1-V4)
- Anteroseptal region
- Bundle branches
- Complications: Heart block, heart failure, cardiogenic shock

**LCx (Left Circumflex):**
- Lateral wall (I, aVL, V5-V6)
- Posterior wall (V7-V9)
- Complications: Mitral regurgitation (papillary muscle dysfunction)

**RCA (Right Coronary Artery):**
- Inferior wall (II, III, aVF)
- Right ventricle (V4R)
- SA node (60%)
- AV node (90%)
- Complications: Bradycardia, heart block, RV failure""",
        "source": "Chou's Electrocardiography in Clinical Practice"
    },
    
    {
        "topic": "Cardiac Biomarkers",
        "content": """**Troponin I/T:**
- Gold standard for MI diagnosis
- Rises: 3-6 hours
- Peaks: 12-24 hours
- Duration: 7-14 days (troponin T longer than I)

**CK-MB:**
- Rises: 4-6 hours
- Peaks: 12-24 hours
- Duration: 2-3 days
- Useful for reinfarction detection

**BNP/NT-proBNP:**
- HF diagnosis and prognosis
- BNP <100 pg/mL: HF unlikely
- BNP >400 pg/mL: HF likely
- NT-proBNP age-adjusted cutoffs

**Myoglobin:**
- Early marker (1-3 hours)
- Poor specificity
- Duration: 12-24 hours""",
        "source": "2023 ESC Guidelines for ACS; Braunwald's Heart Disease"
    }
]


# ============================================
# MAIN FUNCTION TO ADD CARDIOLOGY DATA
# ============================================

def add_cardiology_data():
    """Add cardiology data to existing vectorstore"""
    
    print("🔧 Adding comprehensive cardiology data to fix failing queries...")
    print("="*60)
    
    # Initialize embeddings (same as before)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Load existing vectorstore
    print("📂 Loading existing vectorstore...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Create documents from cardiology Q&As
    documents = []
    
    # Add Q&A pairs (each becomes 2 documents: Q+A combined, and Answer only)
    print(f"📝 Processing {len(CARDIOLOGY_QA)} cardiology Q&As...")
    
    for i, item in enumerate(CARDIOLOGY_QA):
        # Document 1: Question + Answer combined (best for retrieval)
        combined_content = f"""Question: {item['question']}

Answer: {item['answer']}"""
        
        doc_combined = Document(
            page_content=combined_content,
            metadata={
                "source": item['source'],
                "specialty": "cardiology",
                "type": "qa_pair_combined",
                "topic": item['question'][:50],
                "id": f"cardio_qa_combined_{i}"
            }
        )
        documents.append(doc_combined)
        
        # Document 2: Answer only (helps with direct queries)
        doc_answer = Document(
            page_content=item['answer'],
            metadata={
                "source": item['source'],
                "specialty": "cardiology",
                "type": "answer_only",
                "topic": item['question'][:50],
                "id": f"cardio_answer_{i}"
            }
        )
        documents.append(doc_answer)
        
        # Document 3: Question only (helps with question matching)
        doc_question = Document(
            page_content=f"Question: {item['question']}",
            metadata={
                "source": item['source'],
                "specialty": "cardiology",
                "type": "question_only",
                "topic": item['question'][:50],
                "id": f"cardio_question_{i}"
            }
        )
        documents.append(doc_question)
    
    # Add general cardiology knowledge
    print(f"📚 Processing {len(ADDITIONAL_CARDIOLOGY)} general cardiology topics...")
    
    for i, item in enumerate(ADDITIONAL_CARDIOLOGY):
        doc = Document(
            page_content=item['content'],
            metadata={
                "source": item['source'],
                "specialty": "cardiology",
                "topic": item['topic'],
                "type": "general_knowledge",
                "id": f"cardio_gen_{i}"
            }
        )
        documents.append(doc)
    
    print(f"📊 Total documents to add: {len(documents)}")
    print("💾 Adding to vectorstore...")
    
    # Add to vectorstore
    vectorstore.add_documents(documents)
    
    # Persist
    vectorstore.persist()
    
    print(f"✅ Successfully added {len(documents)} cardiology documents!")
    print("="*60)
    
    return len(documents)


# ============================================
# VERIFICATION FUNCTION
# ============================================

def verify_cardiology_fix():
    """Test if cardiology queries now work"""
    
    from src.retrieval.retriever import HybridRetriever
    
    print("\n🔍 VERIFYING CARDIOLOGY FIXES")
    print("="*60)
    
    retriever = HybridRetriever()
    
    test_queries = [
        "PARADIGM-HF study LCZ696 combination sacubitril valsartan",
        "digitalis verapamil contraindicated WPW Wolff Parkinson White",
        "anterior STEMI management chest pain ECG V1-V4",
        "heart rate effects on valvular lesions murmurs diastolic",
        "ECG criteria for STEMI localization LAD RCA",
        "NYHA classification heart failure stages",
        "Vaughan Williams antiarrhythmic drugs classification"
    ]
    
    results_summary = []
    
    for query in test_queries:
        print(f"\n📌 Query: {query}")
        results = retriever.retrieve(query, k=2)
        
        for i, (doc, score) in enumerate(results):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:150].replace('\n', ' ') + "..."
            print(f"  [{i+1}] Score: {score:.4f} | Source: {source}")
            
        # Store result
        if results:
            results_summary.append({
                "query": query[:30] + "...",
                "top_score": results[0][1],
                "found": True
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    for item in results_summary:
        status = "✅" if item['top_score'] > 1.0 else "⚠️"
        print(f"{status} Score: {item['top_score']:.2f} | {item['query']}")
    
    return results_summary


# ============================================
# STEMI-SPECIFIC VERIFICATION
# ============================================

def verify_stemi_fix():
    """Specifically verify STEMI queries are working"""
    
    from src.retrieval.retriever import HybridRetriever
    
    print("\n🔍 VERIFYING STEMI QUERIES (Q10 FIX)")
    print("="*60)
    
    retriever = HybridRetriever()
    
    stemi_queries = [
        "How do you manage acute anterior STEMI?",
        "ECG shows ST elevation in V1-V4 diagnosis and treatment",
        "What is the door-to-balloon time for STEMI?",
        "MONA protocol for myocardial infarction",
        "Complications of anterior wall MI"
    ]
    
    for query in stemi_queries:
        print(f"\n📌 Query: {query}")
        results = retriever.retrieve(query, k=1)
        
        if results:
            doc, score = results[0]
            print(f"  ✅ Score: {score:.4f}")
            print(f"  📄 Preview: {doc.page_content[:200].replace(chr(10), ' ')}...")
        else:
            print(f"  ❌ No results found")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Add cardiology data
    count = add_cardiology_data()
    
    # Verify the fix
    verify_cardiology_fix()
    
    # Specifically verify STEMI
    verify_stemi_fix()
    
    print("\n" + "="*60)
    print("✅ CARDIOLOGY FIX COMPLETE!")
    print("="*60)
    print("\n🎯 Next steps:")
    print("1. Run: python compare_ret.py  (to see improved scores)")
    print("2. Test with: python q2.py")
    print("3. Ask these questions:")
    print("   - 'How do you manage acute anterior STEMI?'")
    print("   - 'What did PARADIGM-HF study show?'")
    print("   - 'Why are digitalis and verapamil contraindicated in WPW?'")
    print("   - 'How does heart rate affect valvular heart lesions?'")