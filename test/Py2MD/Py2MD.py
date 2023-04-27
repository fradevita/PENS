file = open('test1.py', 'r')
file2 = open('converted.md', 'w')
lines = file.readlines()

code_block = False
comment_block = False
for line in lines:
    if (line[:-1] == "'''") and comment_block == False:
        comment_block = True
        if code_block:
            file2.write("```\n")
            code_block = False
        continue
    if line[:-1] == "'''" and comment_block == True:
        comment_block = False
        code_block = True
        file2.write("``` python\n")
        continue
    file2.write(line)
