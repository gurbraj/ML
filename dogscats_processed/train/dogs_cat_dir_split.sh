echo "hello world"

# i="0"
#
# while [ $i -lt 100 ]
# do
# echo $i
# i=$[$i +1]
# done

# if (($i == 1000))
# then
#   echo "exiting"
#   exit 1
# fi
# done

# files= ls
# for file in $files; do
#   echo $file
# done
#cat_files= find . -name '[cat.]*' -exec process {} \;
cat_files= find . -name "[cat.]*"
cat_length= $cat_files | wc -l
# echo "cat_length" $cat_length

i="0"
for cat in cat_files; do
  echo "cat $cat"
  i=$[$i + 1]
  echo "$i"
  # if (($i == $cat_length/10))
  # then
  #   echo "exiting, moved" $i "cats to different folder"
  #   #exit 1
  # else
  #   echo "CAT"
  #   mv $cat ../valid/cats
  # fi
done



# dog_files= find . -name "[dog.]*"
# dog_length= $dog_files | wc -l
#
# echo "dog_length" $dog_length


#worked to move.   {} is the file found
#find . -name "[cat.]*" -exec mv {} cats \;

#from the directory/Users/gk/Desktop/code_july/fast_ai/lesson1/dogscats_unprocessed/train / CATS, use below to populate valid folder
find . -name "[cat.]*" | head -n 1000 -exec mv {} ../../valid/cats \;

find . -name "[dog.]*" | head -n 1000 -exec mv {} ../../valid/dogs \;

#did almost work
find . -name "[cat.]*" | head -1 | xargs -I file mv file ../../valid/cats
find . -name "[dog.]*" | head -1000 | xargs -I file mv file ../../valid/dogs

#arrange the sample
find . -name "[cat.]*" | head -20 | xargs -I file cp file ../../sample/valid/cats
find . -name "[dog.]*" | head -20 | xargs -I file cp file ../../sample/valid/dogs

#making a minor mistake in sample in that i'm putting the same data in valid and training.
