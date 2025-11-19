Use branches: en guide av lukas från it-gruppen

# Kontrollera att du börjar på "main" branchen
git checkout <"mainbranchens" namn>

# Viktigt att börja utveckla från den senaste versionen av dev branchen
git pull origin <"mainbranchens" namn>

# Skapa utvecklingsbranchen och byt till den
git checkout -b <den nya utvecklingsbranchens namn> dev
# exempelvis git checkout -b newbutton dev

# Pusha branchen till github och sätter upstream branchen
git push --set-upstream origin <utvecklingsbranchens namn>

# Utveckla funktionen (gör de ändringar du vill göra i filerna)

# Lägg till filer som ändrats
git add <filnamn1> <filnamn2>
#Alternativ för att lägga till alla filer
git add . 

# Skapa en commit med dina ändringar
git commit -m"commit message"

# Pushar ändringarna till gitlab
git push

# Om du inte tidigare specifierat upstream branch behöver specifiera vilken branch 
# du ska pusha till
git push origin <utvecklingsbranchens namn> 
