import csv
import nltk
#nltk.download('punkt')
from nltk import sent_tokenize,word_tokenize

def breakEssay(essay):
    l=[]
    curr=""
    words=word_tokenize(essay)
    ct=0
    for word in words:
        ct+=1
        if word=="." or word=="!" or word=="?" or ct%50==0:
            if len(curr)==0:
                curr=""
                ct=0
                continue
            curr+=word
            l.append(curr)
            curr=""
            ct=0
            continue
        curr+=word+" "
    return l
add=(breakEssay("i feel cold, my toes are really. brittle feeling, i want to get in my bed right! now, it is so warm in there, i feel closest? to home went i am under my covers trying to sleep, the best feeling ever in the world is being really cold and then jumping into a warm bed that is what i want right now, if my new dorm wasn't so cold i wouldn't be rambling about this topic but wow is it cold although i do enjoy is freezing at night during the day i wouldn't mind a little heat i mean come on now at camp all summer it is so warm i just want to burst every time i walked outside i got that amazing warm sensation over my whole body i remember it so clearly that thawing feeling well i who'll get my hopes up here i am now in college in the cold away from friends and family and camp but i shouldn't feel so isolated sitting in this cold and unhomely room well i should this place is absolutely depressing ha-ha that is so funny how in class we talk about depression and more sleep is associated with it and i totally know that i will be napping after this assignment but no really college is the time of your life or so i am told but so far my two weeks at ut have not been the greatest of my life but then again i have herd stories about people coming here and not liking it for awhile and then they magically start to have fun this is not a cry for help by the way you know the one you warned us about not writing because it would not be read well that is not what this is i am just thing about me and how different i am it is strange the qualities i posses i always think that i am unique and different but there are a million people out there one has to be like me but the more i think about myself the more i am convinced that there is absolutely no one else like me on one side i feel lonely but realistically i love my friends because of the differences between us i feel like all i have is me with your best friends you can tell them anything in the world about how you are feeling and you can trust them and now all those people are gone and when i talk to them on the phone it is just not the same and we seem to be growing a greater distance apart wow i just realized what a sob story this is this is not a cry for help at all what roommate just walked in and totally broke the sob story train i mean he is a nice guy and all and we will get alone fine and everything for a whole year but i definitely don't think that we will be best friends or anything and i really am not going to make a conscience effort to befriend anybody i am quite satisfied all by my self in this way i am really different i mean yes all people need to belong to the group and have friends and fit in but i think that i just have i stronger resistance to such social things as that besides a lot of people have told me that they were never friends with their roommates but that they did have a lot of fun with them so maybe i should be more open to the idea of hanging out with the guy even though he reminds me of a phony awe holden cawfield i can't say phony without thinking about holden cawfield from catcher in the rye i really shouldn't have read that book at the age i did it turned me on in weird ways and i don't think i understood"))
for i in add:
    print(len(word_tokenize(i)))
#exit()
mx=0
m=0
f1 = open('testOPN1.csv','w')
f2 = open('trainOPN1.csv','w')
wr1 = csv.writer(f1, dialect='excel')
wr2 = csv.writer(f2, dialect='excel')
with open('testOPN.csv') as csvfile:
    rdr = csv.reader(csvfile, delimiter=',')
    
    for row in rdr:
        sentences = breakEssay(row[0])
        for sent in sentences:
            m=max(m,len(word_tokenize(sent)))
        sentences += [' '] * (324 - len(sentences))
        mx=max(mx,len(sentences))
        sentences+=[str(row[1])]
        sentences+=[row[2]]
        sentences+=[str(row[3])]
        wr1.writerow(sentences)
        

with open('trainOPN.csv') as csvfile:
    rdr = csv.reader(csvfile, delimiter=',')
    
    for row in rdr:
        sentences = breakEssay(row[0])
        for sent in sentences:
            m=max(m,len(word_tokenize(sent)))
        mx=max(mx,len(sentences))
        sentences += [''] * (324 - len(sentences))
        sentences+=[str(row[1])]
        sentences+=[row[2]]
        sentences+=[str(row[3])]
        wr2.writerow(sentences)
        

print(m)


#max is 324