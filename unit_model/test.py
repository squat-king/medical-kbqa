import re
str='(感冒)-[:症状 {}]->(流鼻涕)'
str_obj=str[:str.index('[')]
str_rel_subj=str[str.index('['):]
obj=re.findall(r"\((.+?)\)",str_obj)
rel=re.findall(r":(.+?) ",str_rel_subj)
subj=re.findall(r"\((.+?)\)",str_rel_subj)
print(obj,rel,subj)
str=".*%s.*"%"感冒"
print(str)

s=['a','b','c']
s_str=""
for _,t in enumerate(s):
    if _ ==0:
        s_str+= t
    else:
        s_str+= ("、"+t)
reply = "{}的{}是{}".format("你","我",s_str)
print(reply)