from memory import MemoryStore

m = MemoryStore(session_id="demo")
m.add_turn("user", "I am flying from Australia.")
m.set_fact("departure_country", "Australia")
print("Facts:", m.get_facts())
print("Recent turns:", m.get_recent_turns())
print("Saved to:", m.path)
