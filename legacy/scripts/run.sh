# 判断输入是否合法
if [ $# -ne 1 ]; then
    echo '请输入: bash scripts/run.sh {name}'
    exit 1
fi

echo '将要运行实验' $1 ':'

echo '编译: g++ main.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm -o bin/'$1
g++ main.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm -o bin/$1

echo '运行: ./bin/'$1
./bin/$1

echo 'git add: git add src/hyperparameter.hpp'
git add src/hyperparameter.hpp -f

echo 'git add: git add bin/'$1 '-f'
git add bin/$1 -f

echo 'git add: git add experiments/'$1'/info.log -f'
git add experiments/$1/info.log -f

echo git add: git ci -m $1
git ci -m $1

echo 'git push'
git push

echo 'python scripts/backup.py --name '$1
python scripts/backup.py --name $1

echo '请注意更新表格!'
