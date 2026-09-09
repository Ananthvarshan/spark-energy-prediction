import os
import zipfile
import glob

def pack_overleaf_project():
    base_dir = r"c:\Users\hydro\OneDrive\Desktop\rp"
    paper_dir = os.path.join(base_dir, "paper")
    zip_path = os.path.join(base_dir, "overleaf_project.zip")
    
    figure_dirs = [
        "outputs/phase7/figures",
        "outputs/phase6/figures",
        "outputs/phase5/figures",
        "outputs/phase4/figures",
        "outputs/phase3/figures/pelletizer-I",
        "outputs/phase2/figures/pelletizer-I"
    ]
    
    print(f"Creating {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        print("Adding and fixing paper files...")
        for root, _, files in os.walk(paper_dir):
            for file in files:
                if file.endswith(('.tex', '.bib')):
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, paper_dir)
                    
                    # Read the file and replace "../outputs" with "outputs"
                    # so Overleaf doesn't get confused
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    content = content.replace("../outputs", "outputs")
                    
                    zipf.writestr(arcname, content)
                    print(f"  Added {arcname}")
                    
        print("\nAdding figure files...")
        for fig_dir in figure_dirs:
            full_fig_dir = os.path.join(base_dir, fig_dir)
            if os.path.exists(full_fig_dir):
                for root, _, files in os.walk(full_fig_dir):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.pdf', '.eps')):
                            full_path = os.path.join(root, file)
                            # NO "../" prefix! Overleaf blocks ZIPs with ../ for security reasons
                            arcname = os.path.relpath(full_path, base_dir).replace('\\', '/')
                            zipf.write(full_path, arcname)
                            print(f"  Added figure {arcname}")
                            
        # We also MUST include the generated ablation table that main.tex imports
        ablation_table_path = os.path.join(base_dir, r"outputs\phase6\task13e\pelletizer-I\ablation_table.tex")
        if os.path.exists(ablation_table_path):
            arcname = "outputs/phase6/task13e/pelletizer-I/ablation_table.tex"
            zipf.write(ablation_table_path, arcname)
            print(f"  Added generated table {arcname}")
                
    print(f"\nDone! You can now upload {zip_path} directly to Overleaf.")

if __name__ == "__main__":
    pack_overleaf_project()
