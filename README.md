# Edit Studio

Εφαρμογή για τον υπολογιστή σου που **δημιουργεί και επεξεργάζεται εικόνες και βίντεο με AI**.
Όλα τρέχουν τοπικά, στις δικές σου κάρτες γραφικών: τίποτα δεν ανεβαίνει στο internet και δεν πληρώνεις
συνδρομή. Το περιβάλλον είναι στα ελληνικά, με επιλογή για αγγλικά.

> English summary at the bottom.

## Τι κάνει

| | |
|---|---|
| 🎨 **Δημιουργία εικόνας** | Γράφεις τι θέλεις και παίρνεις φωτογραφία ή εικονογράφηση. Προτεινόμενα μοντέλα: Z-Image Turbo, FLUX.2 klein, Qwen-Image, SDXL, καθώς και τα δικά σου checkpoints και LoRA. |
| 🪄 **Επεξεργασία εικόνας** | **Με οδηγίες** («κάνε τον ουρανό ηλιοβασίλεμα»), **ζωγράφισε & άλλαξε** (inpaint), **γόμα** αντικειμένων, **επέκταση** κάδρου, **παραλλαγές**, **αφαίρεση φόντου**, **μεγέθυνση** ×2/×4. |
| 🎬 **Δημιουργία βίντεο** | Βίντεο από κείμενο ή «ζωντάνεμα» φωτογραφίας με τα Wan 2.2 και Wan 2.1. |
| ✂️ **Επεξεργασία βίντεο** | **Αλλαγή στυλ** με AI, **συνέχεια** βίντεο με AI, **μεγέθυνση** ανάλυσης, **αφαίρεση φόντου**, **ομαλή κίνηση / slow-motion**, κόψιμο, ταχύτητα, ανάποδα, περικοπή σε 9:16, **GIF**, ένωση clip, εξαγωγή καρέ. |
| 🖼️ **Βιβλιοθήκη** | Όλα τα αποτελέσματα με τις ρυθμίσεις τους, αγαπημένα, σύγκριση πριν/μετά, «ξανά με τις ίδιες ρυθμίσεις». |
| 📦 **Μοντέλα** | **Βρίσκει μόνη της τα μοντέλα που ήδη έχεις** σε ComfyUI, Forge, A1111, Fooocus, StabilityMatrix και στο Hugging Face cache, ώστε να μην τα κατεβάζεις ξανά. |

### Δύο κάρτες γραφικών

Η εφαρμογή βλέπει όλες τις κάρτες NVIDIA και τις χρησιμοποιεί έξυπνα:

* **Δύο δουλειές μαζί.** Π.χ. φτιάχνεις εικόνες στη μία κάρτα ενώ ένα βίντεο «ψήνεται» στην άλλη.
* **Ένα μεγάλο μοντέλο μοιρασμένο και στις δύο.** Μοντέλα πολύ μεγάλα για 16GB (Qwen-Image-Edit 20B, Wan 2.2 A14B)
  μοιράζονται αυτόματα: π.χ. ένας «ειδικός» του Wan 2.2 σε κάθε κάρτα. Όταν χρειάζεται, συμπιέζονται σε 8-bit ή 4-bit.
* Για κάθε μοντέλο υπάρχει έτοιμο σχέδιο μνήμης, και φαίνεται δίπλα στο όνομά του (π.χ. «1 κάρτα + RAM»,
  «Μοιρασμένο σε 2 κάρτες · συμπίεση 4-bit»). Αν τελειώσει η μνήμη, ξαναπροσπαθεί μόνη της με πιο οικονομικό τρόπο.

## Εγκατάσταση (Windows 10/11)

Χρειάζεσαι:

* κάρτα **NVIDIA RTX** με πρόσφατο driver. Για RTX 50xx χρειάζεται driver 580 ή νεότερος:
  [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx)
* **32GB RAM ή περισσότερη** για τα μεγάλα μοντέλα
* **χώρο στον δίσκο**: κάθε μοντέλο πιάνει 7–70GB

Βήματα:

1. Κατέβασε τον κώδικα, είτε από το πράσινο κουμπί **Code → Download ZIP** και αποσυμπίεση, είτε με `git clone`.
   Βάλ' τον κατά προτίμηση σε δίσκο με χώρο, π.χ. `D:\Edit-Studio`.
2. Κάνε διπλό κλικ στο **`install.bat`**. Θα εγκαταστήσει μόνο του:
   * το Python 3.12, απομονωμένα, χωρίς να πειράξει άλλα προγράμματα σου
   * το PyTorch για την κάρτα σου
   * όλα τα υπόλοιπα

   Αν τα Windows πουν «Windows protected your PC», πάτα **More info → Run anyway**.
3. Άνοιξε το **Edit Studio** από τη συντόμευση στην επιφάνεια εργασίας. Αν θέλεις να βλέπεις μηνύματα, άνοιξε το
   `EditStudio.bat`.

Για ενημέρωση σε νεότερη έκδοση: **`update.bat`**.

<details>
<summary>macOS / Linux</summary>

```bash
./install.sh   # μία φορά
./run.sh       # εκκίνηση (ανοίγει στον browser)
```
</details>

## Πρώτα βήματα

1. Στη σελίδα **📦 Μοντέλα** θα δεις τα μοντέλα που βρέθηκαν στον υπολογιστή σου. Αν το ComfyUI ή το Forge σου είναι σε
   ασυνήθιστο φάκελο, πρόσθεσέ τον στο «Φάκελοι».
2. Στη **🎨 Δημιουργία εικόνας** διάλεξε μοντέλο, γράψε περιγραφή και πάτα **Δημιουργία** (ή `Ctrl+Enter`).
   Αν το μοντέλο δεν έχει κατέβει, θα κατέβει αυτόματα. Την πρόοδο τη βλέπεις στις **Εργασίες**, δεξιά.
3. Πάτα πάνω σε ένα αποτέλεσμα για επόμενες κινήσεις: επεξεργασία, «κάν' το βίντεο», μεγέθυνση, αφαίρεση φόντου.

**Προτεινόμενη αρχή για 2 × 16GB:**
* εικόνες με **Z-Image Turbo**
* επεξεργασία με **Qwen-Image-Edit 2511** ή **FLUX.2 klein**
* βίντεο με **Wan 2.2 TI2V 5B**
* για την καλύτερη κίνηση από φωτογραφία, **Wan 2.2 I2V A14B**, μοιρασμένο στις 2 κάρτες

## Μοντέλα και άδειες

| Μοντέλο | Τι κάνει | Άδεια |
|---|---|---|
| Z-Image Turbo | εικόνες, inpaint, παραλλαγές | Apache 2.0 |
| FLUX.2 [klein] 4B | εικόνες + επεξεργασία με αναφορές | Apache 2.0 |
| Qwen-Image 2512 / Qwen-Image-Edit 2511 | εικόνες με κείμενο / επεξεργασία με οδηγίες | Apache 2.0 |
| Qwen-Image 2.1 | εικόνες + επεξεργασία έως 10 αναφορών | Qwen Research (μη εμπορική) |
| Stable Diffusion XL (+ τα δικά σου checkpoints/LoRA) | εικόνες | OpenRAIL++ |
| FLUX.1 Kontext [dev] | επεξεργασία με οδηγίες | FLUX.1 dev (μη εμπορική, θέλει αποδοχή στο HF) |
| Wan 2.2 TI2V 5B, Wan 2.2 A14B, Wan 2.1 1.3B | βίντεο | Apache 2.0 |
| Real-ESRGAN, BiRefNet, LaMa | μεγέθυνση, φόντο, γόμα | BSD / MIT / Apache 2.0 |

Τα μοντέλα με «θέλει αποδοχή» χρειάζονται τρία βήματα:
1. Λογαριασμό στο [huggingface.co](https://huggingface.co).
2. Αποδοχή της άδειας στη σελίδα του μοντέλου.
3. Ένα token στις **⚙️ Ρυθμίσεις**.

### Τα δικά σου αρχεία (ComfyUI / Forge / A1111 / Fooocus)

* ✅ **Checkpoints SD 1.5 / SDXL**, μαζί με Pony, Illustrious, Juggernaut, RealVis και inpainting εκδόσεις. Φορτώνονται απευθείας.
* ✅ **LoRA** για SDXL / SD 1.5 / FLUX.1, με ρύθμιση έντασης.
* ✅ **Μοντέλα μεγέθυνσης**: ESRGAN, RealESRGAN, SwinIR, DAT, HAT κ.λπ.
* ✅ **Αρχεία transformer** FLUX.1, FLUX.2 klein, Z-Image, Qwen-Image και Wan σε bf16, fp16, fp8 ή **GGUF**.
  Τα υπόλοιπα κομμάτια (text encoder, VAE) κατεβαίνουν μία φορά από το Hugging Face, αν δεν τα έχεις ήδη.
* ⚠️ Τα αρχεία **«fp8_scaled»** του ComfyUI δεν διαβάζονται ακόμα. Προτίμησε τις εκδόσεις bf16 ή GGUF.

## Πού αποθηκεύονται τα αρχεία

* **Αποτελέσματα**: `data\outputs\` μέσα στον φάκελο της εφαρμογής. Υπάρχει και κουμπί «Άνοιγμα φακέλου».
* **Μοντέλα Hugging Face**: στον κοινό φάκελο `C:\Users\<όνομα>\.cache\huggingface`, που τον μοιράζονται όλα τα
  προγράμματα. Για να μπαίνουν σε άλλο δίσκο, άλλαξέ τον στις Ρυθμίσεις.
* **Αρχείο καταγραφής**, για όταν κάτι δεν πάει καλά: `data\logs\edit-studio.log`.

## Συχνά προβλήματα

* **«Δεν έφτασε η μνήμη της κάρτας γραφικών»**: Δοκίμασε μικρότερη ανάλυση ή διάρκεια, ή
  **Ρυθμίσεις → Συμπίεση → 4-bit**. Ελευθέρωσε μνήμη από **Μοντέλα → Άδειασμα μνήμης** και κλείσε άλλα προγράμματα
  που χρησιμοποιούν την κάρτα, όπως το ComfyUI.
* **Το μοντέλο θέλει άδεια**: Αποδέξου την άδεια στο huggingface.co και βάλε token στις Ρυθμίσεις.
* **Το PyTorch δεν βλέπει την κάρτα**: Ενημέρωσε τον driver της NVIDIA και ξανατρέξε το `install.bat`.
* **Θέλω να το ανοίξω από κινητό/tablet στο σπίτι**: `EditStudio.bat --listen`. Μόνο σε δίκτυο που εμπιστεύεσαι,
  γιατί τότε η προστασία είναι πιο χαλαρή.

## Για προγραμματιστές

```bash
pip install -e ".[dev]"
pytest            # ~40 τεστ: σχέδια μνήμης, σάρωση μοντέλων, ουρά 2 GPU, εργαλεία βίντεο,
                  # και πραγματικά pipelines diffusers με μικροσκοπικά τυχαία μοντέλα (CPU)
python -m edit_studio --browser --verbose
```

Δομή του κώδικα:

| Αρχείο / φάκελος | Περιεχόμενο |
|---|---|
| `edit_studio/app.py` | API (FastAPI) και ασφάλεια |
| `edit_studio/jobs.py` | ουρά εργασιών: μία «θέση» ανά κάρτα, κοινές δουλειές σε 2 κάρτες |
| `edit_studio/models/catalog.py` | κατάλογος μοντέλων, οικογένειες, pipelines |
| `edit_studio/models/planner.py` | σχέδιο μνήμης: 1 κάρτα / offload / μοίρασμα / συμπίεση |
| `edit_studio/models/manager.py` | φόρτωση με diffusers, cache, LoRA, αλλαγή λειτουργίας με `from_pipe` |
| `edit_studio/models/local.py` | εύρεση και αναγνώριση αρχείων από ComfyUI/Forge/A1111/Fooocus |
| `edit_studio/tasks/` | όλες οι λειτουργίες (εικόνα, βίντεο, εργαλεία) |
| `edit_studio/web/` | το περιβάλλον: απλή JavaScript, χωρίς build |

---

## English

**Edit Studio** is a local desktop app for generating and editing images and video with open AI models
(diffusers-based). Everything runs on your own GPUs.

* **Images**: text-to-image, instruction editing, inpaint, eraser, outpaint, variations, background removal, upscaling.
* **Video**: text-to-video, image-to-video, AI restyle, AI continuation, upscaling, background removal,
  frame interpolation, cut/speed/crop, GIF, join.
* **Models**: finds the models you already have in ComfyUI, Forge, A1111, Fooocus, StabilityMatrix or the HF cache
  (checkpoints, transformer files including GGUF, LoRAs, upscalers).
* **Multi-GPU**: runs two jobs in parallel (one per GPU) or splits one big model over both GPUs (e.g. one Wan 2.2
  A14B expert per card), with automatic fp8/NF4 compression and an out-of-memory fallback.

Install on Windows: run `install.bat`, then open the *Edit Studio* shortcut. On macOS/Linux use `./install.sh` and
`./run.sh`.
