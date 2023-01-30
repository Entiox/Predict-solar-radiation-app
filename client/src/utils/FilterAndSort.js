export default function filterAndSort(array) {
    return array.filter((value, index, self) => self.findIndex(element => element.value === value.value) === index)
        .sort((first, second) => { if(first.value < second.value) return -1; else if(first.value > second.value) return 1; else return 0; })
}