public class IdGenerator {
    private int nextId = 0;

    public IdGenerator() {}
    public IdGenerator(int nextId) {
        this.nextId = nextId;
    }

    public int getNextId() {
        return nextId++;
    }
}
