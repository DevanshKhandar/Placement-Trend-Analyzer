with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
removed_open = 0
removed_close = 0

for line in lines:
    if "st.markdown('<div class=\"glass-card\">', unsafe_allow_html=True)" in line:
        removed_open += 1
        continue
    elif "st.markdown('</div>', unsafe_allow_html=True)" in line:
        removed_close += 1
        continue
    else:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Successfully removed {removed_open} opening div tags and {removed_close} closing div tags.")
